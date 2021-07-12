package golapack

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dtgsyl solves the generalized Sylvester equation:
//
//             A * R - L * B = scale * C                 (1)
//             D * R - L * E = scale * F
//
// where R and L are unknown m-by-n matrices, (A, D), (B, E) and
// (C, F) are given matrix pairs of size m-by-m, n-by-n and m-by-n,
// respectively, with real entries. (A, D) and (B, E) must be in
// generalized (real) Schur canonical form, i.e. A, B are upper quasi
// triangular and D, E are upper triangular.
//
// The solution (R, L) overwrites (C, F). 0 <= SCALE <= 1 is an output
// scaling factor chosen to avoid overflow.
//
// In matrix notation (1) is equivalent to solve  Zx = scale b, where
// Z is defined as
//
//            Z = [ kron(In, A)  -kron(B**T, Im) ]         (2)
//                [ kron(In, D)  -kron(E**T, Im) ].
//
// Here Ik is the identity matrix of size k and X**T is the transpose of
// X. kron(X, Y) is the Kronecker product between the matrices X and Y.
//
// If TRANS = 'T', DTGSYL solves the transposed system Z**T*y = scale*b,
// which is equivalent to solve for R and L in
//
//             A**T * R + D**T * L = scale * C           (3)
//             R * B**T + L * E**T = scale * -F
//
// This case (TRANS = 'T') is used to compute an one-norm-based estimate
// of Dif[(A,D), (B,E)], the separation between the matrix pairs (A,D)
// and (B,E), using DLACON.
//
// If IJOB >= 1, DTGSYL computes a Frobenius norm-based estimate
// of Dif[(A,D),(B,E)]. That is, the reciprocal of a lower bound on the
// reciprocal of the smallest singular value of Z. See [1-2] for more
// information.
//
// This is a level 3 BLAS algorithm.
func Dtgsyl(trans byte, ijob, m, n *int, a *mat.Matrix, lda *int, b *mat.Matrix, ldb *int, c *mat.Matrix, ldc *int, d *mat.Matrix, ldd *int, e *mat.Matrix, lde *int, f *mat.Matrix, ldf *int, scale, dif *float64, work *mat.Vector, lwork *int, iwork *[]int, info *int) {
	var lquery, notran bool
	var dscale, dsum, one, scale2, scaloc, zero float64
	var i, ie, ifunc, iround, is, isolve, j, je, js, k, linfo, lwmin, mb, nb, p, ppqq, pq, q int
	var err error
	_ = err

	zero = 0.0
	one = 1.0

	//     Decode and test input parameters
	(*info) = 0
	notran = trans == 'N'
	lquery = ((*lwork) == -1)

	if !notran && trans != 'T' {
		(*info) = -1
	} else if notran {
		if ((*ijob) < 0) || ((*ijob) > 4) {
			(*info) = -2
		}
	}
	if (*info) == 0 {
		if (*m) <= 0 {
			(*info) = -3
		} else if (*n) <= 0 {
			(*info) = -4
		} else if (*lda) < max(1, *m) {
			(*info) = -6
		} else if (*ldb) < max(1, *n) {
			(*info) = -8
		} else if (*ldc) < max(1, *m) {
			(*info) = -10
		} else if (*ldd) < max(1, *m) {
			(*info) = -12
		} else if (*lde) < max(1, *n) {
			(*info) = -14
		} else if (*ldf) < max(1, *m) {
			(*info) = -16
		}
	}

	if (*info) == 0 {
		if notran {
			if (*ijob) == 1 || (*ijob) == 2 {
				lwmin = max(1, 2*(*m)*(*n))
			} else {
				lwmin = 1
			}
		} else {
			lwmin = 1
		}
		work.Set(0, float64(lwmin))

		if (*lwork) < lwmin && !lquery {
			(*info) = -20
		}
	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("DTGSYL"), -(*info))
		return
	} else if lquery {
		return
	}

	//     Quick return if possible
	if (*m) == 0 || (*n) == 0 {
		(*scale) = 1
		if notran {
			if (*ijob) != 0 {
				(*dif) = 0
			}
		}
		return
	}

	//     Determine optimal block sizes MB and NB
	mb = Ilaenv(func() *int { y := 2; return &y }(), []byte("DTGSYL"), []byte{trans}, m, n, toPtr(-1), toPtr(-1))
	nb = Ilaenv(func() *int { y := 5; return &y }(), []byte("DTGSYL"), []byte{trans}, m, n, toPtr(-1), toPtr(-1))

	isolve = 1
	ifunc = 0
	if notran {
		if (*ijob) >= 3 {
			ifunc = (*ijob) - 2
			Dlaset('F', m, n, &zero, &zero, c, ldc)
			Dlaset('F', m, n, &zero, &zero, f, ldf)
		} else if (*ijob) >= 1 {
			isolve = 2
		}
	}

	if (mb <= 1 && nb <= 1) || (mb >= (*m) && nb >= (*n)) {

		for iround = 1; iround <= isolve; iround++ {
			//           Use unblocked Level 2 solver
			dscale = zero
			dsum = one
			pq = 0
			Dtgsy2(trans, &ifunc, m, n, a, lda, b, ldb, c, ldc, d, ldd, e, lde, f, ldf, scale, &dsum, &dscale, iwork, &pq, info)
			if dscale != zero {
				if (*ijob) == 1 || (*ijob) == 3 {
					(*dif) = math.Sqrt(float64(2*(*m)*(*n))) / (dscale * math.Sqrt(dsum))
				} else {
					(*dif) = math.Sqrt(float64(pq)) / (dscale * math.Sqrt(dsum))
				}
			}

			if isolve == 2 && iround == 1 {
				if notran {
					ifunc = (*ijob)
				}
				scale2 = (*scale)
				Dlacpy('F', m, n, c, ldc, work.Matrix(*m, opts), m)
				Dlacpy('F', m, n, f, ldf, work.MatrixOff((*m)*(*n), *m, opts), m)
				Dlaset('F', m, n, &zero, &zero, c, ldc)
				Dlaset('F', m, n, &zero, &zero, f, ldf)
			} else if isolve == 2 && iround == 2 {
				Dlacpy('F', m, n, work.Matrix(*m, opts), m, c, ldc)
				Dlacpy('F', m, n, work.MatrixOff((*m)*(*n), *m, opts), m, f, ldf)
				(*scale) = scale2
			}
		}

		return
	}

	//     Determine block structure of A
	p = 0
	i = 1
label40:
	;
	if i > (*m) {
		goto label50
	}
	p = p + 1
	(*iwork)[p-1] = i
	i = i + mb
	if i >= (*m) {
		goto label50
	}
	if a.Get(i-1, i-1-1) != zero {
		i = i + 1
	}
	goto label40
label50:
	;

	(*iwork)[p] = (*m) + 1
	if (*iwork)[p-1] == (*iwork)[p] {
		p = p - 1
	}

	//     Determine block structure of B
	q = p + 1
	j = 1
label60:
	;
	if j > (*n) {
		goto label70
	}
	q = q + 1
	(*iwork)[q-1] = j
	j = j + nb
	if j >= (*n) {
		goto label70
	}
	if b.Get(j-1, j-1-1) != zero {
		j = j + 1
	}
	goto label60
label70:
	;

	(*iwork)[q] = (*n) + 1
	if (*iwork)[q-1] == (*iwork)[q] {
		q = q - 1
	}

	if notran {

		for iround = 1; iround <= isolve; iround++ {
			//           Solve (I, J)-subsystem
			//               A(I, I) * R(I, J) - L(I, J) * B(J, J) = C(I, J)
			//               D(I, I) * R(I, J) - L(I, J) * E(J, J) = F(I, J)
			//           for I = P, P - 1,..., 1; J = 1, 2,..., Q
			dscale = zero
			dsum = one
			pq = 0
			(*scale) = one
			for j = p + 2; j <= q; j++ {
				js = (*iwork)[j-1]
				je = (*iwork)[j] - 1
				nb = je - js + 1
				for i = p; i >= 1; i-- {
					is = (*iwork)[i-1]
					ie = (*iwork)[i] - 1
					mb = ie - is + 1
					ppqq = 0
					Dtgsy2(trans, &ifunc, &mb, &nb, a.Off(is-1, is-1), lda, b.Off(js-1, js-1), ldb, c.Off(is-1, js-1), ldc, d.Off(is-1, is-1), ldd, e.Off(js-1, js-1), lde, f.Off(is-1, js-1), ldf, &scaloc, &dsum, &dscale, toSlice(iwork, q+2-1), &ppqq, &linfo)
					if linfo > 0 {
						(*info) = linfo
					}

					pq = pq + ppqq
					if scaloc != one {
						for k = 1; k <= js-1; k++ {
							goblas.Dscal(*m, scaloc, c.Vector(0, k-1, 1))
							goblas.Dscal(*m, scaloc, f.Vector(0, k-1, 1))
						}
						for k = js; k <= je; k++ {
							goblas.Dscal(is-1, scaloc, c.Vector(0, k-1, 1))
							goblas.Dscal(is-1, scaloc, f.Vector(0, k-1, 1))
						}
						for k = js; k <= je; k++ {
							goblas.Dscal((*m)-ie, scaloc, c.Vector(ie, k-1, 1))
							goblas.Dscal((*m)-ie, scaloc, f.Vector(ie, k-1, 1))
						}
						for k = je + 1; k <= (*n); k++ {
							goblas.Dscal(*m, scaloc, c.Vector(0, k-1, 1))
							goblas.Dscal(*m, scaloc, f.Vector(0, k-1, 1))
						}
						(*scale) = (*scale) * scaloc
					}

					//                 Substitute R(I, J) and L(I, J) into remaining
					//                 equation.
					if i > 1 {
						err = goblas.Dgemm(NoTrans, NoTrans, is-1, nb, mb, -one, a.Off(0, is-1), c.Off(is-1, js-1), one, c.Off(0, js-1))
						err = goblas.Dgemm(NoTrans, NoTrans, is-1, nb, mb, -one, d.Off(0, is-1), c.Off(is-1, js-1), one, f.Off(0, js-1))
					}
					if j < q {
						err = goblas.Dgemm(NoTrans, NoTrans, mb, (*n)-je, nb, one, f.Off(is-1, js-1), b.Off(js-1, je), one, c.Off(is-1, je))
						err = goblas.Dgemm(NoTrans, NoTrans, mb, (*n)-je, nb, one, f.Off(is-1, js-1), e.Off(js-1, je), one, f.Off(is-1, je))
					}
				}
			}
			if dscale != zero {
				if (*ijob) == 1 || (*ijob) == 3 {
					(*dif) = math.Sqrt(float64(2*(*m)*(*n))) / (dscale * math.Sqrt(dsum))
				} else {
					(*dif) = math.Sqrt(float64(pq)) / (dscale * math.Sqrt(dsum))
				}
			}
			if isolve == 2 && iround == 1 {
				if notran {
					ifunc = (*ijob)
				}
				scale2 = (*scale)
				Dlacpy('F', m, n, c, ldc, work.Matrix(*m, opts), m)
				Dlacpy('F', m, n, f, ldf, work.MatrixOff((*m)*(*n), *m, opts), m)
				Dlaset('F', m, n, &zero, &zero, c, ldc)
				Dlaset('F', m, n, &zero, &zero, f, ldf)
			} else if isolve == 2 && iround == 2 {
				Dlacpy('F', m, n, work.Matrix(*m, opts), m, c, ldc)
				Dlacpy('F', m, n, work.MatrixOff((*m)*(*n), *m, opts), m, f, ldf)
				(*scale) = scale2
			}
		}

	} else {
		//        Solve transposed (I, J)-subsystem
		//             A(I, I)**T * R(I, J)  + D(I, I)**T * L(I, J)  =  C(I, J)
		//             R(I, J)  * B(J, J)**T + L(I, J)  * E(J, J)**T = -F(I, J)
		//        for I = 1,2,..., P; J = Q, Q-1,..., 1
		(*scale) = one
		for i = 1; i <= p; i++ {
			is = (*iwork)[i-1]
			ie = (*iwork)[i] - 1
			mb = ie - is + 1
			for j = q; j >= p+2; j-- {
				js = (*iwork)[j-1]
				je = (*iwork)[j] - 1
				nb = je - js + 1
				Dtgsy2(trans, &ifunc, &mb, &nb, a.Off(is-1, is-1), lda, b.Off(js-1, js-1), ldb, c.Off(is-1, js-1), ldc, d.Off(is-1, is-1), ldd, e.Off(js-1, js-1), lde, f.Off(is-1, js-1), ldf, &scaloc, &dsum, &dscale, toSlice(iwork, q+2-1), &ppqq, &linfo)
				if linfo > 0 {
					(*info) = linfo
				}
				if scaloc != one {
					for k = 1; k <= js-1; k++ {
						goblas.Dscal(*m, scaloc, c.Vector(0, k-1, 1))
						goblas.Dscal(*m, scaloc, f.Vector(0, k-1, 1))
					}
					for k = js; k <= je; k++ {
						goblas.Dscal(is-1, scaloc, c.Vector(0, k-1, 1))
						goblas.Dscal(is-1, scaloc, f.Vector(0, k-1, 1))
					}
					for k = js; k <= je; k++ {
						goblas.Dscal((*m)-ie, scaloc, c.Vector(ie, k-1, 1))
						goblas.Dscal((*m)-ie, scaloc, f.Vector(ie, k-1, 1))
					}
					for k = je + 1; k <= (*n); k++ {
						goblas.Dscal(*m, scaloc, c.Vector(0, k-1, 1))
						goblas.Dscal(*m, scaloc, f.Vector(0, k-1, 1))
					}
					(*scale) = (*scale) * scaloc
				}

				//              Substitute R(I, J) and L(I, J) into remaining equation.
				if j > p+2 {
					err = goblas.Dgemm(NoTrans, Trans, mb, js-1, nb, one, c.Off(is-1, js-1), b.Off(0, js-1), one, f.Off(is-1, 0))
					err = goblas.Dgemm(NoTrans, Trans, mb, js-1, nb, one, f.Off(is-1, js-1), e.Off(0, js-1), one, f.Off(is-1, 0))
				}
				if i < p {
					err = goblas.Dgemm(Trans, NoTrans, (*m)-ie, nb, mb, -one, a.Off(is-1, ie), c.Off(is-1, js-1), one, c.Off(ie, js-1))
					err = goblas.Dgemm(Trans, NoTrans, (*m)-ie, nb, mb, -one, d.Off(is-1, ie), f.Off(is-1, js-1), one, c.Off(ie, js-1))
				}
			}
		}

	}

	work.Set(0, float64(lwmin))
}
