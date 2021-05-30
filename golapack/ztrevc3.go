package golapack

import (
	"golinalg/goblas"
	"golinalg/golapack/gltest"
	"golinalg/mat"
)

// Ztrevc3 computes some or all of the right and/or left eigenvectors of
// a complex upper triangular matrix T.
// Matrices of this _type are produced by the Schur factorization of
// a complex general matrix:  A = Q*T*Q**H, as computed by ZHSEQR.
//
// The right eigenvector x and the left eigenvector y of T corresponding
// to an eigenvalue w are defined by:
//
//              T*x = w*x,     (y**H)*T = w*(y**H)
//
// where y**H denotes the conjugate transpose of the vector y.
// The eigenvalues are not input to this routine, but are read directly
// from the diagonal of T.
//
// This routine returns the matrices X and/or Y of right and left
// eigenvectors of T, or the products Q*X and/or Q*Y, where Q is an
// input matrix. If Q is the unitary factor that reduces a matrix A to
// Schur form T, then Q*X and Q*Y are the matrices of right and left
// eigenvectors of A.
//
// This uses a Level 3 BLAS version of the back transformation.
func Ztrevc3(side, howmny byte, _select []bool, n *int, t *mat.CMatrix, ldt *int, vl *mat.CMatrix, ldvl *int, vr *mat.CMatrix, ldvr, mm, m *int, work *mat.CVector, lwork *int, rwork *mat.Vector, lrwork, info *int) {
	var allv, bothv, leftv, lquery, over, rightv, somev bool
	var cone, czero complex128
	var one, ovfl, remax, scale, smin, smlnum, ulp, unfl, zero float64
	var i, ii, is, iv, j, k, ki, maxwrk, nb, nbmax, nbmin int

	zero = 0.0
	one = 1.0
	czero = (0.0 + 0.0*1i)
	cone = (1.0 + 0.0*1i)
	nbmin = 8
	nbmax = 128

	//     Decode and test the input parameters
	bothv = side == 'B'
	rightv = side == 'R' || bothv
	leftv = side == 'L' || bothv

	allv = howmny == 'A'
	over = howmny == 'B'
	somev = howmny == 'S'

	//     Set M to the number of columns required to store the selected
	//     eigenvectors.
	if somev {
		(*m) = 0
		for j = 1; j <= (*n); j++ {
			if _select[j-1] {
				(*m) = (*m) + 1
			}
		}
	} else {
		(*m) = (*n)
	}

	(*info) = 0
	nb = Ilaenv(func() *int { y := 1; return &y }(), []byte("ZTREVC"), []byte{side, howmny}, n, toPtr(-1), toPtr(-1), toPtr(-1))
	maxwrk = (*n) + 2*(*n)*nb
	work.SetRe(0, float64(maxwrk))
	rwork.Set(0, float64(*n))
	lquery = ((*lwork) == -1 || (*lrwork) == -1)
	if !rightv && !leftv {
		(*info) = -1
	} else if !allv && !over && !somev {
		(*info) = -2
	} else if (*n) < 0 {
		(*info) = -4
	} else if (*ldt) < maxint(1, *n) {
		(*info) = -6
	} else if (*ldvl) < 1 || (leftv && (*ldvl) < (*n)) {
		(*info) = -8
	} else if (*ldvr) < 1 || (rightv && (*ldvr) < (*n)) {
		(*info) = -10
	} else if (*mm) < (*m) {
		(*info) = -11
	} else if (*lwork) < maxint(1, 2*(*n)) && !lquery {
		(*info) = -14
	} else if (*lrwork) < maxint(1, *n) && !lquery {
		(*info) = -16
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("ZTREVC3"), -(*info))
		return
	} else if lquery {
		return
	}

	//     Quick return if possible.
	if (*n) == 0 {
		return
	}

	//     Use blocked version of back-transformation if sufficient workspace.
	//     Zero-out the workspace to avoid potential NaN propagation.
	if over && (*lwork) >= (*n)+2*(*n)*nbmin {
		nb = ((*lwork) - (*n)) / (2 * (*n))
		nb = minint(nb, nbmax)
		Zlaset('F', n, toPtr(1+2*nb), &czero, &czero, work.CMatrix(*n, opts), n)
	} else {
		nb = 1
	}

	//     Set the constants to control overflow.
	unfl = Dlamch(SafeMinimum)
	ovfl = one / unfl
	Dlabad(&unfl, &ovfl)
	ulp = Dlamch(Precision)
	smlnum = unfl * (float64(*n) / ulp)

	//     Store the diagonal elements of T in working array WORK.
	for i = 1; i <= (*n); i++ {
		work.Set(i-1, t.Get(i-1, i-1))
	}

	//     Compute 1-norm of each column of strictly upper triangular
	//     part of T to control overflow in triangular solver.
	rwork.Set(0, zero)
	for j = 2; j <= (*n); j++ {
		rwork.Set(j-1, goblas.Dzasum(toPtr(j-1), t.CVector(0, j-1), func() *int { y := 1; return &y }()))
	}

	if rightv {
		//        ============================================================
		//        Compute right eigenvectors.
		//
		//        IV is index of column in current block.
		//        Non-blocked version always uses IV=NB=1;
		//        blocked     version starts with IV=NB, goes down to 1.
		//        (Note the "0-th" column is used to store the original diagonal.)
		iv = nb
		is = (*m)
		for ki = (*n); ki >= 1; ki-- {
			if somev {
				if !_select[ki-1] {
					goto label80
				}
			}
			smin = maxf64(ulp*cabs1(t.Get(ki-1, ki-1)), smlnum)

			//           --------------------------------------------------------
			//           Complex right eigenvector
			work.Set(ki+iv*(*n)-1, cone)

			//           Form right-hand side.
			for k = 1; k <= ki-1; k++ {
				work.Set(k+iv*(*n)-1, -t.Get(k-1, ki-1))
			}

			//           Solve upper triangular system:
			//           [ T(1:KI-1,1:KI-1) - T(KI,KI) ]*X = SCALE*WORK.
			for k = 1; k <= ki-1; k++ {
				t.Set(k-1, k-1, t.Get(k-1, k-1)-t.Get(ki-1, ki-1))
				if cabs1(t.Get(k-1, k-1)) < smin {
					t.SetRe(k-1, k-1, smin)
				}
			}

			if ki > 1 {
				Zlatrs('U', 'N', 'N', 'Y', toPtr(ki-1), t, ldt, work.Off(1+iv*(*n)-1), &scale, rwork, info)
				work.SetRe(ki+iv*(*n)-1, scale)
			}

			//           Copy the vector x or Q*x to VR and normalize.
			if !over {
				//              ------------------------------
				//              no back-transform: copy x to VR and normalize.
				goblas.Zcopy(&ki, work.Off(1+iv*(*n)-1), func() *int { y := 1; return &y }(), vr.CVector(0, is-1), func() *int { y := 1; return &y }())
				//
				ii = goblas.Izamax(&ki, vr.CVector(0, is-1), func() *int { y := 1; return &y }())
				remax = one / cabs1(vr.Get(ii-1, is-1))
				goblas.Zdscal(&ki, &remax, vr.CVector(0, is-1), func() *int { y := 1; return &y }())

				for k = ki + 1; k <= (*n); k++ {
					vr.Set(k-1, is-1, czero)
				}

			} else if nb == 1 {
				//              ------------------------------
				//              version 1: back-transform each vector with GEMV, Q*x.
				if ki > 1 {
					goblas.Zgemv(NoTrans, n, toPtr(ki-1), &cone, vr, ldvr, work.Off(1+iv*(*n)-1), func() *int { y := 1; return &y }(), toPtrc128(complex(scale, 0)), vr.CVector(0, ki-1), func() *int { y := 1; return &y }())
				}

				ii = goblas.Izamax(n, vr.CVector(0, ki-1), func() *int { y := 1; return &y }())
				remax = one / cabs1(vr.Get(ii-1, ki-1))
				goblas.Zdscal(n, &remax, vr.CVector(0, ki-1), func() *int { y := 1; return &y }())

			} else {
				//              ------------------------------
				//              version 2: back-transform block of vectors with GEMM
				//              zero out below vector
				for k = ki + 1; k <= (*n); k++ {
					work.Set(k+iv*(*n)-1, czero)
				}

				//              Columns IV:NB of work are valid vectors.
				//              When the number of vectors stored reaches NB,
				//              or if this was last vector, do the GEMM
				if (iv == 1) || (ki == 1) {
					goblas.Zgemm(NoTrans, NoTrans, n, toPtr(nb-iv+1), toPtr(ki+nb-iv), &cone, vr, ldvr, work.CMatrixOff(1+iv*(*n)-1, *n, opts), n, &czero, work.CMatrixOff(1+(nb+iv)*(*n)-1, *n, opts), n)
					//                 normalize vectors
					for k = iv; k <= nb; k++ {
						ii = goblas.Izamax(n, work.Off(1+(nb+k)*(*n)-1), func() *int { y := 1; return &y }())
						remax = one / cabs1(work.Get(ii+(nb+k)*(*n)-1))
						goblas.Zdscal(n, &remax, work.Off(1+(nb+k)*(*n)-1), func() *int { y := 1; return &y }())
					}
					Zlacpy('F', n, toPtr(nb-iv+1), work.CMatrixOff(1+(nb+iv)*(*n)-1, *n, opts), n, vr.Off(0, ki-1), ldvr)
					iv = nb
				} else {
					iv = iv - 1
				}
			}

			//           Restore the original diagonal elements of T.
			for k = 1; k <= ki-1; k++ {
				t.Set(k-1, k-1, work.Get(k-1))
			}

			is = is - 1
		label80:
		}
	}

	if leftv {
		//        ============================================================
		//        Compute left eigenvectors.
		//
		//        IV is index of column in current block.
		//        Non-blocked version always uses IV=1;
		//        blocked     version starts with IV=1, goes up to NB.
		//        (Note the "0-th" column is used to store the original diagonal.)
		iv = 1
		is = 1
		for ki = 1; ki <= (*n); ki++ {
			//
			if somev {
				if !_select[ki-1] {
					goto label130
				}
			}
			smin = maxf64(ulp*cabs1(t.Get(ki-1, ki-1)), smlnum)

			//           --------------------------------------------------------
			//           Complex left eigenvector
			work.Set(ki+iv*(*n)-1, cone)

			//           Form right-hand side.
			for k = ki + 1; k <= (*n); k++ {
				work.Set(k+iv*(*n)-1, -t.GetConj(ki-1, k-1))
			}

			//           Solve conjugate-transposed triangular system:
			//           [ T(KI+1:N,KI+1:N) - T(KI,KI) ]**H * X = SCALE*WORK.
			for k = ki + 1; k <= (*n); k++ {
				t.Set(k-1, k-1, t.Get(k-1, k-1)-t.Get(ki-1, ki-1))
				if cabs1(t.Get(k-1, k-1)) < smin {
					t.SetRe(k-1, k-1, smin)
				}
			}

			if ki < (*n) {
				Zlatrs('U', 'C', 'N', 'Y', toPtr((*n)-ki), t.Off(ki+1-1, ki+1-1), ldt, work.Off(ki+1+iv*(*n)-1), &scale, rwork, info)
				work.SetRe(ki+iv*(*n)-1, scale)
			}

			//           Copy the vector x or Q*x to VL and normalize.
			if !over {
				//              ------------------------------
				//              no back-transform: copy x to VL and normalize.
				goblas.Zcopy(toPtr((*n)-ki+1), work.Off(ki+iv*(*n)-1), func() *int { y := 1; return &y }(), vl.CVector(ki-1, is-1), func() *int { y := 1; return &y }())
				//
				ii = goblas.Izamax(toPtr((*n)-ki+1), vl.CVector(ki-1, is-1), func() *int { y := 1; return &y }()) + ki - 1
				remax = one / cabs1(vl.Get(ii-1, is-1))
				goblas.Zdscal(toPtr((*n)-ki+1), &remax, vl.CVector(ki-1, is-1), func() *int { y := 1; return &y }())
				//
				for k = 1; k <= ki-1; k++ {
					vl.Set(k-1, is-1, czero)
				}

			} else if nb == 1 {
				//              ------------------------------
				//              version 1: back-transform each vector with GEMV, Q*x.
				if ki < (*n) {
					goblas.Zgemv(NoTrans, n, toPtr((*n)-ki), &cone, vl.Off(0, ki+1-1), ldvl, work.Off(ki+1+iv*(*n)-1), func() *int { y := 1; return &y }(), toPtrc128(complex(scale, 0)), vl.CVector(0, ki-1), func() *int { y := 1; return &y }())
				}

				ii = goblas.Izamax(n, vl.CVector(0, ki-1), func() *int { y := 1; return &y }())
				remax = one / cabs1(vl.Get(ii-1, ki-1))
				goblas.Zdscal(n, &remax, vl.CVector(0, ki-1), func() *int { y := 1; return &y }())

			} else {
				//              ------------------------------
				//              version 2: back-transform block of vectors with GEMM
				//              zero out above vector
				//              could go from KI-NV+1 to KI-1
				for k = 1; k <= ki-1; k++ {
					work.Set(k+iv*(*n)-1, czero)
				}

				//              Columns 1:IV of work are valid vectors.
				//              When the number of vectors stored reaches NB,
				//              or if this was last vector, do the GEMM
				if (iv == nb) || (ki == (*n)) {
					goblas.Zgemm(NoTrans, NoTrans, n, &iv, toPtr((*n)-ki+iv), &cone, vl.Off(0, ki-iv+1-1), ldvl, work.CMatrixOff(ki-iv+1+1*(*n)-1, *n, opts), n, &czero, work.CMatrixOff(1+(nb+1)*(*n)-1, *n, opts), n)
					//                 normalize vectors
					for k = 1; k <= iv; k++ {
						ii = goblas.Izamax(n, work.Off(1+(nb+k)*(*n)-1), func() *int { y := 1; return &y }())
						remax = one / cabs1(work.Get(ii+(nb+k)*(*n)-1))
						goblas.Zdscal(n, &remax, work.Off(1+(nb+k)*(*n)-1), func() *int { y := 1; return &y }())
					}
					Zlacpy('F', n, &iv, work.CMatrixOff(1+(nb+1)*(*n)-1, *n, opts), n, vl.Off(0, ki-iv+1-1), ldvl)
					iv = 1
				} else {
					iv = iv + 1
				}
			}

			//           Restore the original diagonal elements of T.
			for k = ki + 1; k <= (*n); k++ {
				t.Set(k-1, k-1, work.Get(k-1))
			}

			is = is + 1
		label130:
		}
	}
}
