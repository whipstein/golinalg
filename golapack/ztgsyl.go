package golapack

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Ztgsyl solves the generalized Sylvester equation:
//
//             A * R - L * B = scale * C            (1)
//             D * R - L * E = scale * F
//
// where R and L are unknown m-by-n matrices, (A, D), (B, E) and
// (C, F) are given matrix pairs of size m-by-m, n-by-n and m-by-n,
// respectively, with complex entries. A, B, D and E are upper
// triangular (i.e., (A,D) and (B,E) in generalized Schur form).
//
// The solution (R, L) overwrites (C, F). 0 <= SCALE <= 1
// is an output scaling factor chosen to avoid overflow.
//
// In matrix notation (1) is equivalent to solve Zx = scale*b, where Z
// is defined as
//
//        Z = [ kron(In, A)  -kron(B**H, Im) ]        (2)
//            [ kron(In, D)  -kron(E**H, Im) ],
//
// Here Ix is the identity matrix of size x and X**H is the conjugate
// transpose of X. Kron(X, Y) is the Kronecker product between the
// matrices X and Y.
//
// If TRANS = 'C', y in the conjugate transposed system Z**H *y = scale*b
// is solved for, which is equivalent to solve for R and L in
//
//             A**H * R + D**H * L = scale * C           (3)
//             R * B**H + L * E**H = scale * -F
//
// This case (TRANS = 'C') is used to compute an one-norm-based estimate
// of Dif[(A,D), (B,E)], the separation between the matrix pairs (A,D)
// and (B,E), using ZLACON.
//
// If IJOB >= 1, ZTGSYL computes a Frobenius norm-based estimate of
// Dif[(A,D),(B,E)]. That is, the reciprocal of a lower bound on the
// reciprocal of the smallest singular value of Z.
//
// This is a level-3 BLAS algorithm.
func Ztgsyl(trans byte, ijob, m, n *int, a *mat.CMatrix, lda *int, b *mat.CMatrix, ldb *int, c *mat.CMatrix, ldc *int, d *mat.CMatrix, ldd *int, e *mat.CMatrix, lde *int, f *mat.CMatrix, ldf *int, scale, dif *float64, work *mat.CVector, lwork *int, iwork *[]int, info *int) {
	var lquery, notran bool
	var czero complex128
	var dscale, dsum, one, scale2, scaloc, zero float64
	var i, ie, ifunc, iround, is, isolve, j, je, js, k, linfo, lwmin, mb, nb, p, pq, q int

	zero = 0.0
	one = 1.0
	czero = (0.0 + 0.0*1i)

	//     Decode and test input parameters
	(*info) = 0
	notran = trans == 'N'
	lquery = ((*lwork) == -1)

	if !notran && trans != 'C' {
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
		} else if (*lda) < maxint(1, *m) {
			(*info) = -6
		} else if (*ldb) < maxint(1, *n) {
			(*info) = -8
		} else if (*ldc) < maxint(1, *m) {
			(*info) = -10
		} else if (*ldd) < maxint(1, *m) {
			(*info) = -12
		} else if (*lde) < maxint(1, *n) {
			(*info) = -14
		} else if (*ldf) < maxint(1, *m) {
			(*info) = -16
		}
	}

	if (*info) == 0 {
		if notran {
			if (*ijob) == 1 || (*ijob) == 2 {
				lwmin = maxint(1, 2*(*m)*(*n))
			} else {
				lwmin = 1
			}
		} else {
			lwmin = 1
		}
		work.SetRe(0, float64(lwmin))

		if (*lwork) < lwmin && !lquery {
			(*info) = -20
		}
	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("ZTGSYL"), -(*info))
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

	//     Determine  optimal block sizes MB and NB
	mb = Ilaenv(func() *int { y := 2; return &y }(), []byte("ZTGSYL"), []byte{trans}, m, n, toPtr(-1), toPtr(-1))
	nb = Ilaenv(func() *int { y := 5; return &y }(), []byte("ZTGSYL"), []byte{trans}, m, n, toPtr(-1), toPtr(-1))

	isolve = 1
	ifunc = 0
	if notran {
		if (*ijob) >= 3 {
			ifunc = (*ijob) - 2
			Zlaset('F', m, n, &czero, &czero, c, ldc)
			Zlaset('F', m, n, &czero, &czero, f, ldf)
		} else if (*ijob) >= 1 && notran {
			isolve = 2
		}
	}

	if (mb <= 1 && nb <= 1) || (mb >= (*m) && nb >= (*n)) {
		//        Use unblocked Level 2 solver
		for iround = 1; iround <= isolve; iround++ {

			(*scale) = one
			dscale = zero
			dsum = one
			pq = (*m) * (*n)
			Ztgsy2(trans, &ifunc, m, n, a, lda, b, ldb, c, ldc, d, ldd, e, lde, f, ldf, scale, &dsum, &dscale, info)
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
				Zlacpy('F', m, n, c, ldc, work.CMatrix(*m, opts), m)
				Zlacpy('F', m, n, f, ldf, work.CMatrixOff((*m)*(*n)+1-1, *m, opts), m)
				Zlaset('F', m, n, &czero, &czero, c, ldc)
				Zlaset('F', m, n, &czero, &czero, f, ldf)
			} else if isolve == 2 && iround == 2 {
				Zlacpy('F', m, n, work.CMatrix(*m, opts), m, c, ldc)
				Zlacpy('F', m, n, work.CMatrixOff((*m)*(*n)+1-1, *m, opts), m, f, ldf)
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
	goto label40
label50:
	;
	(*iwork)[p+1-1] = (*m) + 1
	if (*iwork)[p-1] == (*iwork)[p+1-1] {
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
	goto label60

label70:
	;
	(*iwork)[q+1-1] = (*n) + 1
	if (*iwork)[q-1] == (*iwork)[q+1-1] {
		q = q - 1
	}

	if notran {
		for iround = 1; iround <= isolve; iround++ {
			//           Solve (I, J) - subsystem
			//               A(I, I) * R(I, J) - L(I, J) * B(J, J) = C(I, J)
			//               D(I, I) * R(I, J) - L(I, J) * E(J, J) = F(I, J)
			//           for I = P, P - 1, ..., 1; J = 1, 2, ..., Q
			pq = 0
			(*scale) = one
			dscale = zero
			dsum = one
			for j = p + 2; j <= q; j++ {
				js = (*iwork)[j-1]
				je = (*iwork)[j+1-1] - 1
				nb = je - js + 1
				for i = p; i >= 1; i-- {
					is = (*iwork)[i-1]
					ie = (*iwork)[i+1-1] - 1
					mb = ie - is + 1
					Ztgsy2(trans, &ifunc, &mb, &nb, a.Off(is-1, is-1), lda, b.Off(js-1, js-1), ldb, c.Off(is-1, js-1), ldc, d.Off(is-1, is-1), ldd, e.Off(js-1, js-1), lde, f.Off(is-1, js-1), ldf, &scaloc, &dsum, &dscale, &linfo)
					if linfo > 0 {
						(*info) = linfo
					}
					pq = pq + mb*nb
					if scaloc != one {
						for k = 1; k <= js-1; k++ {
							goblas.Zscal(m, toPtrc128(complex(scaloc, zero)), c.CVector(0, k-1), func() *int { y := 1; return &y }())
							goblas.Zscal(m, toPtrc128(complex(scaloc, zero)), f.CVector(0, k-1), func() *int { y := 1; return &y }())
						}
						for k = js; k <= je; k++ {
							goblas.Zscal(toPtr(is-1), toPtrc128(complex(scaloc, zero)), c.CVector(0, k-1), func() *int { y := 1; return &y }())
							goblas.Zscal(toPtr(is-1), toPtrc128(complex(scaloc, zero)), f.CVector(0, k-1), func() *int { y := 1; return &y }())
						}
						for k = js; k <= je; k++ {
							goblas.Zscal(toPtr((*m)-ie), toPtrc128(complex(scaloc, zero)), c.CVector(ie+1-1, k-1), func() *int { y := 1; return &y }())
							goblas.Zscal(toPtr((*m)-ie), toPtrc128(complex(scaloc, zero)), f.CVector(ie+1-1, k-1), func() *int { y := 1; return &y }())
						}
						for k = je + 1; k <= (*n); k++ {
							goblas.Zscal(m, toPtrc128(complex(scaloc, zero)), c.CVector(0, k-1), func() *int { y := 1; return &y }())
							goblas.Zscal(m, toPtrc128(complex(scaloc, zero)), f.CVector(0, k-1), func() *int { y := 1; return &y }())
						}
						(*scale) = (*scale) * scaloc
					}

					//                 Substitute R(I,J) and L(I,J) into remaining equation.
					if i > 1 {
						goblas.Zgemm(NoTrans, NoTrans, toPtr(is-1), &nb, &mb, toPtrc128(complex(-one, zero)), a.Off(0, is-1), lda, c.Off(is-1, js-1), ldc, toPtrc128(complex(one, zero)), c.Off(0, js-1), ldc)
						goblas.Zgemm(NoTrans, NoTrans, toPtr(is-1), &nb, &mb, toPtrc128(complex(-one, zero)), d.Off(0, is-1), ldd, c.Off(is-1, js-1), ldc, toPtrc128(complex(one, zero)), f.Off(0, js-1), ldf)
					}
					if j < q {
						goblas.Zgemm(NoTrans, NoTrans, &mb, toPtr((*n)-je), &nb, toPtrc128(complex(one, zero)), f.Off(is-1, js-1), ldf, b.Off(js-1, je+1-1), ldb, toPtrc128(complex(one, zero)), c.Off(is-1, je+1-1), ldc)
						goblas.Zgemm(NoTrans, NoTrans, &mb, toPtr((*n)-je), &nb, toPtrc128(complex(one, zero)), f.Off(is-1, js-1), ldf, e.Off(js-1, je+1-1), lde, toPtrc128(complex(one, zero)), f.Off(is-1, je+1-1), ldf)
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
				Zlacpy('F', m, n, c, ldc, work.CMatrix(*m, opts), m)
				Zlacpy('F', m, n, f, ldf, work.CMatrixOff((*m)*(*n)+1-1, *m, opts), m)
				Zlaset('F', m, n, &czero, &czero, c, ldc)
				Zlaset('F', m, n, &czero, &czero, f, ldf)
			} else if isolve == 2 && iround == 2 {
				Zlacpy('F', m, n, work.CMatrix(*m, opts), m, c, ldc)
				Zlacpy('F', m, n, work.CMatrixOff((*m)*(*n)+1-1, *m, opts), m, f, ldf)
				(*scale) = scale2
			}
		}
	} else {
		//        Solve transposed (I, J)-subsystem
		//            A(I, I)**H * R(I, J) + D(I, I)**H * L(I, J) = C(I, J)
		//            R(I, J) * B(J, J)  + L(I, J) * E(J, J) = -F(I, J)
		//        for I = 1,2,..., P; J = Q, Q-1,..., 1
		(*scale) = one
		for i = 1; i <= p; i++ {
			is = (*iwork)[i-1]
			ie = (*iwork)[i+1-1] - 1
			mb = ie - is + 1
			for j = q; j >= p+2; j-- {
				js = (*iwork)[j-1]
				je = (*iwork)[j+1-1] - 1
				nb = je - js + 1
				Ztgsy2(trans, &ifunc, &mb, &nb, a.Off(is-1, is-1), lda, b.Off(js-1, js-1), ldb, c.Off(is-1, js-1), ldc, d.Off(is-1, is-1), ldd, e.Off(js-1, js-1), lde, f.Off(is-1, js-1), ldf, &scaloc, &dsum, &dscale, &linfo)
				if linfo > 0 {
					(*info) = linfo
				}
				if scaloc != one {
					for k = 1; k <= js-1; k++ {
						goblas.Zscal(m, toPtrc128(complex(scaloc, zero)), c.CVector(0, k-1), func() *int { y := 1; return &y }())
						goblas.Zscal(m, toPtrc128(complex(scaloc, zero)), f.CVector(0, k-1), func() *int { y := 1; return &y }())
					}
					for k = js; k <= je; k++ {
						goblas.Zscal(toPtr(is-1), toPtrc128(complex(scaloc, zero)), c.CVector(0, k-1), func() *int { y := 1; return &y }())
						goblas.Zscal(toPtr(is-1), toPtrc128(complex(scaloc, zero)), f.CVector(0, k-1), func() *int { y := 1; return &y }())
					}
					for k = js; k <= je; k++ {
						goblas.Zscal(toPtr((*m)-ie), toPtrc128(complex(scaloc, zero)), c.CVector(ie+1-1, k-1), func() *int { y := 1; return &y }())
						goblas.Zscal(toPtr((*m)-ie), toPtrc128(complex(scaloc, zero)), f.CVector(ie+1-1, k-1), func() *int { y := 1; return &y }())
					}
					for k = je + 1; k <= (*n); k++ {
						goblas.Zscal(m, toPtrc128(complex(scaloc, zero)), c.CVector(0, k-1), func() *int { y := 1; return &y }())
						goblas.Zscal(m, toPtrc128(complex(scaloc, zero)), f.CVector(0, k-1), func() *int { y := 1; return &y }())
					}
					(*scale) = (*scale) * scaloc
				}

				//              Substitute R(I,J) and L(I,J) into remaining equation.
				if j > p+2 {
					goblas.Zgemm(NoTrans, ConjTrans, &mb, toPtr(js-1), &nb, toPtrc128(complex(one, zero)), c.Off(is-1, js-1), ldc, b.Off(0, js-1), ldb, toPtrc128(complex(one, zero)), f.Off(is-1, 0), ldf)
					goblas.Zgemm(NoTrans, ConjTrans, &mb, toPtr(js-1), &nb, toPtrc128(complex(one, zero)), f.Off(is-1, js-1), ldf, e.Off(0, js-1), lde, toPtrc128(complex(one, zero)), f.Off(is-1, 0), ldf)
				}
				if i < p {
					goblas.Zgemm(ConjTrans, NoTrans, toPtr((*m)-ie), &nb, &mb, toPtrc128(complex(-one, zero)), a.Off(is-1, ie+1-1), lda, c.Off(is-1, js-1), ldc, toPtrc128(complex(one, zero)), c.Off(ie+1-1, js-1), ldc)
					goblas.Zgemm(ConjTrans, NoTrans, toPtr((*m)-ie), &nb, &mb, toPtrc128(complex(-one, zero)), d.Off(is-1, ie+1-1), ldd, f.Off(is-1, js-1), ldf, toPtrc128(complex(one, zero)), c.Off(ie+1-1, js-1), ldc)
				}
			}
		}
	}

	work.SetRe(0, float64(lwmin))
}
