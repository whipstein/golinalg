package golapack

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dtgsna estimates reciprocal condition numbers for specified
// eigenvalues and/or eigenvectors of a matrix pair (A, B) in
// generalized real Schur canonical form (or of any matrix pair
// (Q*A*Z**T, Q*B*Z**T) with orthogonal matrices Q and Z, where
// Z**T denotes the transpose of Z.
//
// (A, B) must be in generalized real Schur form (as returned by DGGES),
// i.e. A is block upper triangular with 1-by-1 and 2-by-2 diagonal
// blocks. B is upper triangular.
func Dtgsna(job, howmny byte, _select []bool, n *int, a *mat.Matrix, lda *int, b *mat.Matrix, ldb *int, vl *mat.Matrix, ldvl *int, vr *mat.Matrix, ldvr *int, s, dif *mat.Vector, mm, m *int, work *mat.Vector, lwork *int, iwork *[]int, info *int) {
	var lquery, pair, somcon, wantbh, wantdf, wants bool
	var alphai, alphar, alprqt, beta, c1, c2, cond, eps, four, lnrm, one, rnrm, root1, root2, scale, smlnum, tmpii, tmpir, tmpri, tmprr, two, uhav, uhavi, uhbv, uhbvi, zero float64
	var difdri, i, ierr, ifst, ilst, iz, k, ks, lwmin, n1, n2 int

	dummy := vf(1)
	dummy1 := vf(1)

	difdri = 3
	zero = 0.0
	one = 1.0
	two = 2.0
	four = 4.0

	//     Decode and test the input parameters
	wantbh = job == 'B'
	wants = job == 'E' || wantbh
	wantdf = job == 'V' || wantbh

	somcon = howmny == 'S'

	(*info) = 0
	lquery = ((*lwork) == -1)

	if !wants && !wantdf {
		(*info) = -1
	} else if howmny != 'A' && !somcon {
		(*info) = -2
	} else if (*n) < 0 {
		(*info) = -4
	} else if (*lda) < maxint(1, *n) {
		(*info) = -6
	} else if (*ldb) < maxint(1, *n) {
		(*info) = -8
	} else if wants && (*ldvl) < (*n) {
		(*info) = -10
	} else if wants && (*ldvr) < (*n) {
		(*info) = -12
	} else {
		//        Set M to the number of eigenpairs for which condition numbers
		//        are required, and test MM.
		if somcon {
			(*m) = 0
			pair = false
			for k = 1; k <= (*n); k++ {
				if pair {
					pair = false
				} else {
					if k < (*n) {
						if a.Get(k+1-1, k-1) == zero {
							if _select[k-1] {
								(*m) = (*m) + 1
							}
						} else {
							pair = true
							if _select[k-1] || _select[k+1-1] {
								(*m) = (*m) + 2
							}
						}
					} else {
						if _select[(*n)-1] {
							(*m) = (*m) + 1
						}
					}
				}
			}
		} else {
			(*m) = (*n)
		}

		if (*n) == 0 {
			lwmin = 1
		} else if job == 'V' || job == 'B' {
			lwmin = 2*(*n)*((*n)+2) + 16
		} else {
			lwmin = (*n)
		}
		work.Set(0, float64(lwmin))

		if (*mm) < (*m) {
			(*info) = -15
		} else if (*lwork) < lwmin && !lquery {
			(*info) = -18
		}
	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("DTGSNA"), -(*info))
		return
	} else if lquery {
		return
	}

	//     Quick return if possible
	if (*n) == 0 {
		return
	}

	//     Get machine constants
	eps = Dlamch(Precision)
	smlnum = Dlamch(SafeMinimum) / eps
	ks = 0
	pair = false

	for k = 1; k <= (*n); k++ {
		//        Determine whether A(k,k) begins a 1-by-1 or 2-by-2 block.
		if pair {
			pair = false
			goto label20
		} else {
			if k < (*n) {
				pair = a.Get(k+1-1, k-1) != zero
			}
		}

		//        Determine whether condition numbers are required for the k-th
		//        eigenpair.
		if somcon {
			if pair {
				if !_select[k-1] && !_select[k+1-1] {
					goto label20
				}
			} else {
				if !_select[k-1] {
					goto label20
				}
			}
		}

		ks = ks + 1

		if wants {
			//           Compute the reciprocal condition number of the k-th
			//           eigenvalue.
			if pair {
				//              Complex eigenvalue pair.
				rnrm = Dlapy2(toPtrf64(goblas.Dnrm2(n, vr.Vector(0, ks-1), func() *int { y := 1; return &y }())), toPtrf64(goblas.Dnrm2(n, vr.Vector(0, ks+1-1), func() *int { y := 1; return &y }())))
				lnrm = Dlapy2(toPtrf64(goblas.Dnrm2(n, vl.Vector(0, ks-1), func() *int { y := 1; return &y }())), toPtrf64(goblas.Dnrm2(n, vl.Vector(0, ks+1-1), func() *int { y := 1; return &y }())))
				goblas.Dgemv(NoTrans, n, n, &one, a, lda, vr.Vector(0, ks-1), func() *int { y := 1; return &y }(), &zero, work, func() *int { y := 1; return &y }())
				tmprr = goblas.Ddot(n, work, func() *int { y := 1; return &y }(), vl.Vector(0, ks-1), func() *int { y := 1; return &y }())
				tmpri = goblas.Ddot(n, work, func() *int { y := 1; return &y }(), vl.Vector(0, ks+1-1), func() *int { y := 1; return &y }())
				goblas.Dgemv(NoTrans, n, n, &one, a, lda, vr.Vector(0, ks+1-1), func() *int { y := 1; return &y }(), &zero, work, func() *int { y := 1; return &y }())
				tmpii = goblas.Ddot(n, work, func() *int { y := 1; return &y }(), vl.Vector(0, ks+1-1), func() *int { y := 1; return &y }())
				tmpir = goblas.Ddot(n, work, func() *int { y := 1; return &y }(), vl.Vector(0, ks-1), func() *int { y := 1; return &y }())
				uhav = tmprr + tmpii
				uhavi = tmpir - tmpri
				goblas.Dgemv(NoTrans, n, n, &one, b, ldb, vr.Vector(0, ks-1), func() *int { y := 1; return &y }(), &zero, work, func() *int { y := 1; return &y }())
				tmprr = goblas.Ddot(n, work, func() *int { y := 1; return &y }(), vl.Vector(0, ks-1), func() *int { y := 1; return &y }())
				tmpri = goblas.Ddot(n, work, func() *int { y := 1; return &y }(), vl.Vector(0, ks+1-1), func() *int { y := 1; return &y }())
				goblas.Dgemv(NoTrans, n, n, &one, b, ldb, vr.Vector(0, ks+1-1), func() *int { y := 1; return &y }(), &zero, work, func() *int { y := 1; return &y }())
				tmpii = goblas.Ddot(n, work, func() *int { y := 1; return &y }(), vl.Vector(0, ks+1-1), func() *int { y := 1; return &y }())
				tmpir = goblas.Ddot(n, work, func() *int { y := 1; return &y }(), vl.Vector(0, ks-1), func() *int { y := 1; return &y }())
				uhbv = tmprr + tmpii
				uhbvi = tmpir - tmpri
				uhav = Dlapy2(&uhav, &uhavi)
				uhbv = Dlapy2(&uhbv, &uhbvi)
				cond = Dlapy2(&uhav, &uhbv)
				s.Set(ks-1, cond/(rnrm*lnrm))
				s.Set(ks+1-1, s.Get(ks-1))

			} else {
				//              Real eigenvalue.
				rnrm = goblas.Dnrm2(n, vr.Vector(0, ks-1), func() *int { y := 1; return &y }())
				lnrm = goblas.Dnrm2(n, vl.Vector(0, ks-1), func() *int { y := 1; return &y }())
				goblas.Dgemv(NoTrans, n, n, &one, a, lda, vr.Vector(0, ks-1), func() *int { y := 1; return &y }(), &zero, work, func() *int { y := 1; return &y }())
				uhav = goblas.Ddot(n, work, func() *int { y := 1; return &y }(), vl.Vector(0, ks-1), func() *int { y := 1; return &y }())
				goblas.Dgemv(NoTrans, n, n, &one, b, ldb, vr.Vector(0, ks-1), func() *int { y := 1; return &y }(), &zero, work, func() *int { y := 1; return &y }())
				uhbv = goblas.Ddot(n, work, func() *int { y := 1; return &y }(), vl.Vector(0, ks-1), func() *int { y := 1; return &y }())
				cond = Dlapy2(&uhav, &uhbv)
				if cond == zero {
					s.Set(ks-1, -one)
				} else {
					s.Set(ks-1, cond/(rnrm*lnrm))
				}
			}
		}

		if wantdf {
			if (*n) == 1 {
				dif.Set(ks-1, Dlapy2(a.GetPtr(0, 0), b.GetPtr(0, 0)))
				goto label20
			}

			//           Estimate the reciprocal condition number of the k-th
			//           eigenvectors.
			if pair {
				//              Copy the  2-by 2 pencil beginning at (A(k,k), B(k, k)).
				//              Compute the eigenvalue(s) at position K.
				work.Set(0, a.Get(k-1, k-1))
				work.Set(1, a.Get(k+1-1, k-1))
				work.Set(2, a.Get(k-1, k+1-1))
				work.Set(3, a.Get(k+1-1, k+1-1))
				work.Set(4, b.Get(k-1, k-1))
				work.Set(5, b.Get(k+1-1, k-1))
				work.Set(6, b.Get(k-1, k+1-1))
				work.Set(7, b.Get(k+1-1, k+1-1))
				Dlag2(work.Matrix(2, opts), func() *int { y := 2; return &y }(), work.MatrixOff(4, 2, opts), func() *int { y := 2; return &y }(), toPtrf64(smlnum*eps), &beta, dummy1.GetPtr(0), &alphar, dummy.GetPtr(0), &alphai)
				alprqt = one
				c1 = two * (alphar*alphar + alphai*alphai + beta*beta)
				c2 = four * beta * beta * alphai * alphai
				root1 = c1 + math.Sqrt(c1*c1-4.0*c2)
				root2 = c2 / root1
				root1 = root1 / two
				cond = minf64(math.Sqrt(root1), math.Sqrt(root2))
			}

			//           Copy the matrix (A, B) to the array WORK and swap the
			//           diagonal block beginning at A(k,k) to the (1,1) position.
			Dlacpy('F', n, n, a, lda, work.Matrix(*n, opts), n)
			Dlacpy('F', n, n, b, ldb, work.MatrixOff((*n)*(*n)+1-1, *n, opts), n)
			ifst = k
			ilst = 1

			Dtgexc(false, false, n, work.Matrix(*n, opts), n, work.MatrixOff((*n)*(*n)+1-1, *n, opts), n, dummy.Matrix(1, opts), func() *int { y := 1; return &y }(), dummy1.Matrix(1, opts), func() *int { y := 1; return &y }(), &ifst, &ilst, work.Off((*n)*(*n)*2+1-1), toPtr((*lwork)-2*(*n)*(*n)), &ierr)

			if ierr > 0 {
				//              Ill-conditioned problem - swap rejected.
				dif.Set(ks-1, zero)
			} else {
				//              Reordering successful, solve generalized Sylvester
				//              equation for R and L,
				//                         A22 * R - L * A11 = A12
				//                         B22 * R - L * B11 = B12,
				//              and compute estimate of Difl((A11,B11), (A22, B22)).
				n1 = 1
				if work.Get(1) != zero {
					n1 = 2
				}
				n2 = (*n) - n1
				if n2 == 0 {
					dif.Set(ks-1, cond)
				} else {
					i = (*n)*(*n) + 1
					iz = 2*(*n)*(*n) + 1
					Dtgsyl('N', &difdri, &n2, &n1, work.MatrixOff((*n)*n1+n1+1-1, *n, opts), n, work.Matrix(*n, opts), n, work.MatrixOff(n1+1-1, *n, opts), n, work.MatrixOff((*n)*n1+n1+i-1, *n, opts), n, work.MatrixOff(i-1, *n, opts), n, work.MatrixOff(n1+i-1, *n, opts), n, &scale, dif.GetPtr(ks-1), work.Off(iz+1-1), toPtr((*lwork)-2*(*n)*(*n)), iwork, &ierr)

					if pair {
						dif.Set(ks-1, minf64(maxf64(one, alprqt)*dif.Get(ks-1), cond))
					}
				}
			}
			if pair {
				dif.Set(ks+1-1, dif.Get(ks-1))
			}
		}
		if pair {
			ks = ks + 1
		}

	label20:
	}
	work.Set(0, float64(lwmin))
}
