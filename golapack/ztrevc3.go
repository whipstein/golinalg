package golapack

import (
	"fmt"
	"math"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
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
func Ztrevc3(side mat.MatSide, howmny byte, _select []bool, n int, t, vl, vr *mat.CMatrix, mm int, work *mat.CVector, lwork int, rwork *mat.Vector, lrwork int) (m int, err error) {
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
	bothv = side == Both
	rightv = side == Right || bothv
	leftv = side == Left || bothv

	allv = howmny == 'A'
	over = howmny == 'B'
	somev = howmny == 'S'

	//     Set M to the number of columns required to store the selected
	//     eigenvectors.
	if somev {
		m = 0
		for j = 1; j <= n; j++ {
			if _select[j-1] {
				m = m + 1
			}
		}
	} else {
		m = n
	}

	nb = Ilaenv(1, "Ztrevc", []byte{side.Byte(), howmny}, n, -1, -1, -1)
	maxwrk = n + 2*n*nb
	work.SetRe(0, float64(maxwrk))
	rwork.Set(0, float64(n))
	lquery = (lwork == -1 || lrwork == -1)
	if !rightv && !leftv {
		err = fmt.Errorf("!rightv && !leftv: side=%s", side)
	} else if !allv && !over && !somev {
		err = fmt.Errorf("!allv && !over && !somev: howmny='%c'", howmny)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if t.Rows < max(1, n) {
		err = fmt.Errorf("t.Rows < max(1, n): t.Rows=%v, n=%v", t.Rows, n)
	} else if vl.Rows < 1 || (leftv && vl.Rows < n) {
		err = fmt.Errorf("vl.Rows < 1 || (leftv && vl.Rows < n): side=%s, vl.Rows=%v, n=%v", side, vl.Rows, n)
	} else if vr.Rows < 1 || (rightv && vr.Rows < n) {
		err = fmt.Errorf("vr.Rows < 1 || (rightv && vr.Rows < n): side=%s, vr.Rows=%v, n=%v", side, vr.Rows, n)
	} else if mm < m {
		err = fmt.Errorf("mm < m: mm=%v, m=%v", mm, m)
	} else if lwork < max(1, 2*n) && !lquery {
		err = fmt.Errorf("lwork < max(1, 2*n) && !lquery: lwork=%v, n=%v, lquery=%v", lwork, n, lquery)
	} else if lrwork < max(1, n) && !lquery {
		err = fmt.Errorf("lrwork < max(1, n) && !lquery: lrwork=%v, n=%v, lquery=%v", lrwork, n, lquery)
	}
	if err != nil {
		gltest.Xerbla2("Ztrevc3", err)
		return
	} else if lquery {
		return
	}

	//     Quick return if possible.
	if n == 0 {
		return
	}

	//     Use blocked version of back-transformation if sufficient workspace.
	//     Zero-out the workspace to avoid potential NaN propagation.
	if over && lwork >= n+2*n*nbmin {
		nb = (lwork - n) / (2 * n)
		nb = min(nb, nbmax)
		Zlaset(Full, n, 1+2*nb, czero, czero, work.CMatrix(n, opts))
	} else {
		nb = 1
	}

	//     Set the constants to control overflow.
	unfl = Dlamch(SafeMinimum)
	ovfl = one / unfl
	unfl, ovfl = Dlabad(unfl, ovfl)
	ulp = Dlamch(Precision)
	smlnum = unfl * (float64(n) / ulp)

	//     Store the diagonal elements of T in working array WORK.
	for i = 1; i <= n; i++ {
		work.Set(i-1, t.Get(i-1, i-1))
	}

	//     Compute 1-norm of each column of strictly upper triangular
	//     part of T to control overflow in triangular solver.
	rwork.Set(0, zero)
	for j = 2; j <= n; j++ {
		rwork.Set(j-1, t.Off(0, j-1).CVector().Asum(j-1, 1))
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
		is = m
		for ki = n; ki >= 1; ki-- {
			if somev {
				if !_select[ki-1] {
					goto label80
				}
			}
			smin = math.Max(ulp*cabs1(t.Get(ki-1, ki-1)), smlnum)

			//           --------------------------------------------------------
			//           Complex right eigenvector
			work.Set(ki+iv*n-1, cone)

			//           Form right-hand side.
			for k = 1; k <= ki-1; k++ {
				work.Set(k+iv*n-1, -t.Get(k-1, ki-1))
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
				if scale, err = Zlatrs(Upper, NoTrans, NonUnit, 'Y', ki-1, t, work.Off(1+iv*n-1), rwork); err != nil {
					panic(err)
				}
				work.SetRe(ki+iv*n-1, scale)
			}

			//           Copy the vector x or Q*x to VR and normalize.
			if !over {
				//              ------------------------------
				//              no back-transform: copy x to VR and normalize.
				vr.Off(0, is-1).CVector().Copy(ki, work.Off(1+iv*n-1), 1, 1)
				//
				ii = vr.Off(0, is-1).CVector().Iamax(ki, 1)
				remax = one / cabs1(vr.Get(ii-1, is-1))
				vr.Off(0, is-1).CVector().Dscal(ki, remax, 1)

				for k = ki + 1; k <= n; k++ {
					vr.Set(k-1, is-1, czero)
				}

			} else if nb == 1 {
				//              ------------------------------
				//              version 1: back-transform each vector with GEMV, Q*x.
				if ki > 1 {
					if err = vr.Off(0, ki-1).CVector().Gemv(NoTrans, n, ki-1, cone, vr, work.Off(1+iv*n-1), 1, complex(scale, 0), 1); err != nil {
						panic(err)
					}
				}

				ii = vr.Off(0, ki-1).CVector().Iamax(n, 1)
				remax = one / cabs1(vr.Get(ii-1, ki-1))
				vr.Off(0, ki-1).CVector().Dscal(n, remax, 1)

			} else {
				//              ------------------------------
				//              version 2: back-transform block of vectors with GEMM
				//              zero out below vector
				for k = ki + 1; k <= n; k++ {
					work.Set(k+iv*n-1, czero)
				}

				//              Columns IV:NB of work are valid vectors.
				//              When the number of vectors stored reaches NB,
				//              or if this was last vector, do the GEMM
				if (iv == 1) || (ki == 1) {
					if err = work.Off(1+(nb+iv)*n-1).CMatrix(n, opts).Gemm(NoTrans, NoTrans, n, nb-iv+1, ki+nb-iv, cone, vr, work.Off(1+iv*n-1).CMatrix(n, opts), czero); err != nil {
						panic(err)
					}
					//                 normalize vectors
					for k = iv; k <= nb; k++ {
						ii = work.Off(1+(nb+k)*n-1).Iamax(n, 1)
						remax = one / cabs1(work.Get(ii+(nb+k)*n-1))
						work.Off(1+(nb+k)*n-1).Dscal(n, remax, 1)
					}
					Zlacpy(Full, n, nb-iv+1, work.Off(1+(nb+iv)*n-1).CMatrix(n, opts), vr.Off(0, ki-1))
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
		for ki = 1; ki <= n; ki++ {
			//
			if somev {
				if !_select[ki-1] {
					goto label130
				}
			}
			smin = math.Max(ulp*cabs1(t.Get(ki-1, ki-1)), smlnum)

			//           --------------------------------------------------------
			//           Complex left eigenvector
			work.Set(ki+iv*n-1, cone)

			//           Form right-hand side.
			for k = ki + 1; k <= n; k++ {
				work.Set(k+iv*n-1, -t.GetConj(ki-1, k-1))
			}

			//           Solve conjugate-transposed triangular system:
			//           [ T(KI+1:N,KI+1:N) - T(KI,KI) ]**H * X = SCALE*WORK.
			for k = ki + 1; k <= n; k++ {
				t.Set(k-1, k-1, t.Get(k-1, k-1)-t.Get(ki-1, ki-1))
				if cabs1(t.Get(k-1, k-1)) < smin {
					t.SetRe(k-1, k-1, smin)
				}
			}

			if ki < n {
				if scale, err = Zlatrs(Upper, ConjTrans, NonUnit, 'Y', n-ki, t.Off(ki, ki), work.Off(ki+1+iv*n-1), rwork); err != nil {
					panic(err)
				}
				work.SetRe(ki+iv*n-1, scale)
			}

			//           Copy the vector x or Q*x to VL and normalize.
			if !over {
				//              ------------------------------
				//              no back-transform: copy x to VL and normalize.
				vl.Off(ki-1, is-1).CVector().Copy(n-ki+1, work.Off(ki+iv*n-1), 1, 1)
				//
				ii = vl.Off(ki-1, is-1).CVector().Iamax(n-ki+1, 1) + ki - 1
				remax = one / cabs1(vl.Get(ii-1, is-1))
				vl.Off(ki-1, is-1).CVector().Dscal(n-ki+1, remax, 1)
				//
				for k = 1; k <= ki-1; k++ {
					vl.Set(k-1, is-1, czero)
				}

			} else if nb == 1 {
				//              ------------------------------
				//              version 1: back-transform each vector with GEMV, Q*x.
				if ki < n {
					if err = vl.Off(0, ki-1).CVector().Gemv(NoTrans, n, n-ki, cone, vl.Off(0, ki), work.Off(ki+1+iv*n-1), 1, complex(scale, 0), 1); err != nil {
						panic(err)
					}
				}

				ii = vl.Off(0, ki-1).CVector().Iamax(n, 1)
				remax = one / cabs1(vl.Get(ii-1, ki-1))
				vl.Off(0, ki-1).CVector().Dscal(n, remax, 1)

			} else {
				//              ------------------------------
				//              version 2: back-transform block of vectors with GEMM
				//              zero out above vector
				//              could go from KI-NV+1 to KI-1
				for k = 1; k <= ki-1; k++ {
					work.Set(k+iv*n-1, czero)
				}

				//              Columns 1:IV of work are valid vectors.
				//              When the number of vectors stored reaches NB,
				//              or if this was last vector, do the GEMM
				if (iv == nb) || (ki == n) {
					if err = work.Off(1+(nb+1)*n-1).CMatrix(n, opts).Gemm(NoTrans, NoTrans, n, iv, n-ki+iv, cone, vl.Off(0, ki-iv), work.Off(ki-iv+1+1*n-1).CMatrix(n, opts), czero); err != nil {
						panic(err)
					}
					//                 normalize vectors
					for k = 1; k <= iv; k++ {
						ii = work.Off(1+(nb+k)*n-1).Iamax(n, 1)
						remax = one / cabs1(work.Get(ii+(nb+k)*n-1))
						work.Off(1+(nb+k)*n-1).Dscal(n, remax, 1)
					}
					Zlacpy(Full, n, iv, work.Off(1+(nb+1)*n-1).CMatrix(n, opts), vl.Off(0, ki-iv))
					iv = 1
				} else {
					iv = iv + 1
				}
			}

			//           Restore the original diagonal elements of T.
			for k = ki + 1; k <= n; k++ {
				t.Set(k-1, k-1, work.Get(k-1))
			}

			is = is + 1
		label130:
		}
	}

	return
}
