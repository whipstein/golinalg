package golapack

import (
	"fmt"
	"math"
	"math/cmplx"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zhetf2rook computes the factorization of a complex Hermitian matrix A
// using the bounded Bunch-Kaufman ("rook") diagonal pivoting method:
//
//    A = U*D*U**H  or  A = L*D*L**H
//
// where U (or L) is a product of permutation and unit upper (lower)
// triangular matrices, U**H is the conjugate transpose of U, and D is
// Hermitian and block diagonal with 1-by-1 and 2-by-2 diagonal blocks.
//
// This is the unblocked version of the algorithm, calling Level 2 BLAS.
func Zhetf2Rook(uplo mat.MatUplo, n int, a *mat.CMatrix, ipiv *[]int) (info int, err error) {
	var done, upper bool
	var d12, d21, t, wk, wkm1, wkp1 complex128
	var absakk, alpha, colmax, d, d11, d22, dtemp, eight, one, r1, rowmax, sevten, sfmin, tt, zero float64
	var i, ii, imax, itemp, j, jmax, k, kk, kp, kstep, p int

	zero = 0.0
	one = 1.0
	eight = 8.0
	sevten = 17.0

	//     Test the input parameters.
	upper = uplo == Upper
	if !upper && uplo != Lower {
		err = fmt.Errorf("!upper && uplo != Lower: uplo=%s", uplo)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if a.Rows < max(1, n) {
		err = fmt.Errorf("a.Rows < max(1, n): a.Rows=%v, n=%v", a.Rows, n)
	}
	if err != nil {
		gltest.Xerbla2("Zhetf2Rook", err)
		return
	}

	//     Initialize ALPHA for use in choosing pivot block size.
	alpha = (one + math.Sqrt(sevten)) / eight

	//     Compute machine safe minimum
	sfmin = Dlamch(SafeMinimum)

	if upper {
		//        Factorize A as U*D*U**H using the upper triangle of A
		//
		//        K is the main loop index, decreasing from N to 1 in steps of
		//        1 or 2
		k = n
	label10:
		;

		//        If K < 1, exit from loop
		if k < 1 {
			return
		}
		kstep = 1
		p = k

		//        Determine rows and columns to be interchanged and whether
		//        a 1-by-1 or 2-by-2 pivot block will be used
		absakk = math.Abs(a.GetRe(k-1, k-1))

		//        IMAX is the row-index of the largest off-diagonal element in
		//        column K, and COLMAX is its absolute value.
		//        Determine both COLMAX and IMAX.
		if k > 1 {
			imax = a.Off(0, k-1).CVector().Iamax(k-1, 1)
			colmax = cabs1(a.Get(imax-1, k-1))
		} else {
			colmax = zero
		}

		if math.Max(absakk, colmax) == zero {
			//           Column K is zero or underflow: set INFO and continue
			if info == 0 {
				info = k
			}
			kp = k
			a.Set(k-1, k-1, a.GetReCmplx(k-1, k-1))
		} else {
			//           ============================================================
			//
			//           BEGIN pivot search
			//
			//           Case(1)
			//           Equivalent to testing for ABSAKK.GE.ALPHA*COLMAX
			//           (used to handle NaN and Inf)
			if !(absakk < alpha*colmax) {
				//              no interchange, use 1-by-1 pivot block
				kp = k

			} else {
				done = false

				//              Loop until pivot found
			label12:
				;

				//                 BEGIN pivot search loop body
				//
				//
				//                 JMAX is the column-index of the largest off-diagonal
				//                 element in row IMAX, and ROWMAX is its absolute value.
				//                 Determine both ROWMAX and JMAX.
				if imax != k {
					jmax = imax + a.Off(imax-1, imax).CVector().Iamax(k-imax, a.Rows)
					rowmax = cabs1(a.Get(imax-1, jmax-1))
				} else {
					rowmax = zero
				}

				if imax > 1 {
					itemp = a.Off(0, imax-1).CVector().Iamax(imax-1, 1)
					dtemp = cabs1(a.Get(itemp-1, imax-1))
					if dtemp > rowmax {
						rowmax = dtemp
						jmax = itemp
					}
				}

				//                 Case(2)
				//                 Equivalent to testing for
				//                 ABS( REAL( W( IMAX,KW-1 ) ) ).GE.ALPHA*ROWMAX
				//                 (used to handle NaN and Inf)
				if !(math.Abs(a.GetRe(imax-1, imax-1)) < alpha*rowmax) {
					//                    interchange rows and columns K and IMAX,
					//                    use 1-by-1 pivot block
					kp = imax
					done = true

					//                 Case(3)
					//                 Equivalent to testing for ROWMAX.EQ.COLMAX,
					//                 (used to handle NaN and Inf)
				} else if (p == jmax) || (rowmax <= colmax) {
					//                    interchange rows and columns K-1 and IMAX,
					//                    use 2-by-2 pivot block
					kp = imax
					kstep = 2
					done = true

					//                 Case(4)
				} else {
					//                    Pivot not found: set params and repeat
					p = imax
					colmax = rowmax
					imax = jmax
				}

				//                 END pivot search loop body
				if !done {
					goto label12
				}

			}

			//           END pivot search
			//
			//           ============================================================
			//
			//           KK is the column of A where pivoting step stopped
			kk = k - kstep + 1

			//           For only a 2x2 pivot, interchange rows and columns K and P
			//           in the leading submatrix A(1:k,1:k)
			if (kstep == 2) && (p != k) {
				//              (1) Swap columnar parts
				if p > 1 {
					a.Off(0, p-1).CVector().Swap(p-1, a.Off(0, k-1).CVector(), 1, 1)
				}
				//              (2) Swap and conjugate middle parts
				for j = p + 1; j <= k-1; j++ {
					t = a.GetConj(j-1, k-1)
					a.Set(j-1, k-1, a.GetConj(p-1, j-1))
					a.Set(p-1, j-1, t)
				}
				//              (3) Swap and conjugate corner elements at row-col interserction
				a.Set(p-1, k-1, a.GetConj(p-1, k-1))
				//              (4) Swap diagonal elements at row-col intersection
				r1 = a.GetRe(k-1, k-1)
				a.Set(k-1, k-1, a.GetReCmplx(p-1, p-1))
				a.SetRe(p-1, p-1, r1)
			}

			//           For both 1x1 and 2x2 pivots, interchange rows and
			//           columns KK and KP in the leading submatrix A(1:k,1:k)
			if kp != kk {
				//              (1) Swap columnar parts
				if kp > 1 {
					a.Off(0, kp-1).CVector().Swap(kp-1, a.Off(0, kk-1).CVector(), 1, 1)
				}
				//              (2) Swap and conjugate middle parts
				for j = kp + 1; j <= kk-1; j++ {
					t = a.GetConj(j-1, kk-1)
					a.Set(j-1, kk-1, a.GetConj(kp-1, j-1))
					a.Set(kp-1, j-1, t)
				}
				//              (3) Swap and conjugate corner elements at row-col interserction
				a.Set(kp-1, kk-1, a.GetConj(kp-1, kk-1))
				//              (4) Swap diagonal elements at row-col intersection
				r1 = a.GetRe(kk-1, kk-1)
				a.Set(kk-1, kk-1, a.GetReCmplx(kp-1, kp-1))
				a.SetRe(kp-1, kp-1, r1)

				if kstep == 2 {
					//                 (*) Make sure that diagonal element of pivot is real
					a.Set(k-1, k-1, a.GetReCmplx(k-1, k-1))
					//                 (5) Swap row elements
					t = a.Get(k-1-1, k-1)
					a.Set(k-1-1, k-1, a.Get(kp-1, k-1))
					a.Set(kp-1, k-1, t)
				}
			} else {
				//              (*) Make sure that diagonal element of pivot is real
				a.Set(k-1, k-1, a.GetReCmplx(k-1, k-1))
				if kstep == 2 {
					a.Set(k-1-1, k-1-1, a.GetReCmplx(k-1-1, k-1-1))
				}
			}

			//           Update the leading submatrix
			if kstep == 1 {
				//              1-by-1 pivot block D(k): column k now holds
				//
				//              W(k) = U(k)*D(k)
				//
				//              where U(k) is the k-th column of U
				if k > 1 {
					//                 Perform a rank-1 update of A(1:k-1,1:k-1) and
					//                 store U(k) in column k
					if math.Abs(a.GetRe(k-1, k-1)) >= sfmin {
						//                    Perform a rank-1 update of A(1:k-1,1:k-1) as
						//                    A := A - U(k)*D(k)*U(k)**T
						//                       = A - W(k)*1/D(k)*W(k)**T
						d11 = one / a.GetRe(k-1, k-1)
						if err = a.Her(uplo, k-1, -d11, a.Off(0, k-1).CVector(), 1); err != nil {
							panic(err)
						}

						//                    Store U(k) in column k
						a.Off(0, k-1).CVector().Dscal(k-1, d11, 1)
					} else {
						//                    Store L(k) in column K
						d11 = a.GetRe(k-1, k-1)
						for ii = 1; ii <= k-1; ii++ {
							a.Set(ii-1, k-1, a.Get(ii-1, k-1)/complex(d11, 0))
						}

						//                    Perform a rank-1 update of A(k+1:n,k+1:n) as
						//                    A := A - U(k)*D(k)*U(k)**T
						//                       = A - W(k)*(1/D(k))*W(k)**T
						//                       = A - (W(k)/D(k))*(D(k))*(W(k)/D(K))**T
						if err = a.Her(uplo, k-1, -d11, a.Off(0, k-1).CVector(), 1); err != nil {
							panic(err)
						}
					}
				}

			} else {
				//              2-by-2 pivot block D(k): columns k and k-1 now hold
				//
				//              ( W(k-1) W(k) ) = ( U(k-1) U(k) )*D(k)
				//
				//              where U(k) and U(k-1) are the k-th and (k-1)-th columns
				//              of U
				//
				//              Perform a rank-2 update of A(1:k-2,1:k-2) as
				//
				//              A := A - ( U(k-1) U(k) )*D(k)*( U(k-1) U(k) )**T
				//                 = A - ( ( A(k-1)A(k) )*inv(D(k)) ) * ( A(k-1)A(k) )**T
				//
				//              and store L(k) and L(k+1) in columns k and k+1
				if k > 2 {
					//                 D = |A12|
					d = Dlapy2(a.GetRe(k-1-1, k-1), a.GetIm(k-1-1, k-1))
					d11 = a.GetRe(k-1, k-1) / d
					d22 = a.GetRe(k-1-1, k-1-1) / d
					d12 = a.Get(k-1-1, k-1) / complex(d, 0)
					tt = one / (d11*d22 - one)

					for j = k - 2; j >= 1; j-- {
						//                    Compute  D21 * ( W(k)W(k+1) ) * inv(D(k)) for row J
						wkm1 = complex(tt, 0) * (complex(d11, 0)*a.Get(j-1, k-1-1) - cmplx.Conj(d12)*a.Get(j-1, k-1))
						wk = complex(tt, 0) * (complex(d22, 0)*a.Get(j-1, k-1) - d12*a.Get(j-1, k-1-1))

						//                    Perform a rank-2 update of A(1:k-2,1:k-2)
						for i = j; i >= 1; i-- {
							a.Set(i-1, j-1, a.Get(i-1, j-1)-(a.Get(i-1, k-1)/complex(d, 0))*cmplx.Conj(wk)-(a.Get(i-1, k-1-1)/complex(d, 0))*cmplx.Conj(wkm1))
						}

						//                    Store U(k) and U(k-1) in cols k and k-1 for row J
						a.Set(j-1, k-1, wk/complex(d, 0))
						a.Set(j-1, k-1-1, wkm1/complex(d, 0))
						//                    (*) Make sure that diagonal element of pivot is real
						a.Set(j-1, j-1, a.GetReCmplx(j-1, j-1))

					}

				}

			}

		}

		//        Store details of the interchanges in IPIV
		if kstep == 1 {
			(*ipiv)[k-1] = kp
		} else {
			(*ipiv)[k-1] = -p
			(*ipiv)[k-1-1] = -kp
		}

		//        Decrease K and return to the start of the main loop
		k = k - kstep
		goto label10

	} else {
		//        Factorize A as L*D*L**H using the lower triangle of A
		//
		//        K is the main loop index, increasing from 1 to N in steps of
		//        1 or 2
		k = 1
	label40:
		;

		//        If K > N, exit from loop
		if k > n {
			return
		}
		kstep = 1
		p = k

		//        Determine rows and columns to be interchanged and whether
		//        a 1-by-1 or 2-by-2 pivot block will be used
		absakk = math.Abs(a.GetRe(k-1, k-1))

		//        IMAX is the row-index of the largest off-diagonal element in
		//        column K, and COLMAX is its absolute value.
		//        Determine both COLMAX and IMAX.
		if k < n {
			imax = k + a.Off(k, k-1).CVector().Iamax(n-k, 1)
			colmax = cabs1(a.Get(imax-1, k-1))
		} else {
			colmax = zero
		}

		if math.Max(absakk, colmax) == zero {
			//           Column K is zero or underflow: set INFO and continue
			if info == 0 {
				info = k
			}
			kp = k
			a.Set(k-1, k-1, a.GetReCmplx(k-1, k-1))
		} else {
			//           ============================================================
			//
			//           BEGIN pivot search
			//
			//           Case(1)
			//           Equivalent to testing for ABSAKK.GE.ALPHA*COLMAX
			//           (used to handle NaN and Inf)
			if !(absakk < alpha*colmax) {
				//              no interchange, use 1-by-1 pivot block
				kp = k

			} else {

				done = false

				//              Loop until pivot found
			label42:
				;

				//                 BEGIN pivot search loop body
				//
				//
				//                 JMAX is the column-index of the largest off-diagonal
				//                 element in row IMAX, and ROWMAX is its absolute value.
				//                 Determine both ROWMAX and JMAX.
				if imax != k {
					jmax = k - 1 + a.Off(imax-1, k-1).CVector().Iamax(imax-k, a.Rows)
					rowmax = cabs1(a.Get(imax-1, jmax-1))
				} else {
					rowmax = zero
				}

				if imax < n {
					itemp = imax + a.Off(imax, imax-1).CVector().Iamax(n-imax, 1)
					dtemp = cabs1(a.Get(itemp-1, imax-1))
					if dtemp > rowmax {
						rowmax = dtemp
						jmax = itemp
					}
				}

				//                 Case(2)
				//                 Equivalent to testing for
				//                 ABS( REAL( W( IMAX,KW-1 ) ) ).GE.ALPHA*ROWMAX
				//                 (used to handle NaN and Inf)
				if !(math.Abs(a.GetRe(imax-1, imax-1)) < alpha*rowmax) {
					//                    interchange rows and columns K and IMAX,
					//                    use 1-by-1 pivot block
					kp = imax
					done = true

					//                 Case(3)
					//                 Equivalent to testing for ROWMAX.EQ.COLMAX,
					//                 (used to handle NaN and Inf)
				} else if (p == jmax) || (rowmax <= colmax) {

					//                    interchange rows and columns K+1 and IMAX,
					//                    use 2-by-2 pivot block
					kp = imax
					kstep = 2
					done = true

					//                 Case(4)
				} else {
					//                    Pivot not found: set params and repeat
					p = imax
					colmax = rowmax
					imax = jmax
				}

				//                 END pivot search loop body
				if !done {
					goto label42
				}

			}

			//           END pivot search
			//
			//           ============================================================
			//
			//           KK is the column of A where pivoting step stopped
			kk = k + kstep - 1

			//           For only a 2x2 pivot, interchange rows and columns K and P
			//           in the trailing submatrix A(k:n,k:n)
			if (kstep == 2) && (p != k) {
				//              (1) Swap columnar parts
				if p < n {
					a.Off(p, p-1).CVector().Swap(n-p, a.Off(p, k-1).CVector(), 1, 1)
				}
				//              (2) Swap and conjugate middle parts
				for j = k + 1; j <= p-1; j++ {
					t = a.GetConj(j-1, k-1)
					a.Set(j-1, k-1, a.GetConj(p-1, j-1))
					a.Set(p-1, j-1, t)
				}
				//              (3) Swap and conjugate corner elements at row-col interserction
				a.Set(p-1, k-1, a.GetConj(p-1, k-1))
				//              (4) Swap diagonal elements at row-col intersection
				r1 = a.GetRe(k-1, k-1)
				a.Set(k-1, k-1, a.GetReCmplx(p-1, p-1))
				a.SetRe(p-1, p-1, r1)
			}

			//           For both 1x1 and 2x2 pivots, interchange rows and
			//           columns KK and KP in the trailing submatrix A(k:n,k:n)
			if kp != kk {
				//              (1) Swap columnar parts
				if kp < n {
					a.Off(kp, kp-1).CVector().Swap(n-kp, a.Off(kp, kk-1).CVector(), 1, 1)
				}
				//              (2) Swap and conjugate middle parts
				for j = kk + 1; j <= kp-1; j++ {
					t = a.GetConj(j-1, kk-1)
					a.Set(j-1, kk-1, a.GetConj(kp-1, j-1))
					a.Set(kp-1, j-1, t)
				}
				//              (3) Swap and conjugate corner elements at row-col interserction
				a.Set(kp-1, kk-1, a.GetConj(kp-1, kk-1))
				//              (4) Swap diagonal elements at row-col intersection
				r1 = a.GetRe(kk-1, kk-1)
				a.Set(kk-1, kk-1, a.GetReCmplx(kp-1, kp-1))
				a.SetRe(kp-1, kp-1, r1)

				if kstep == 2 {
					//                 (*) Make sure that diagonal element of pivot is real
					a.Set(k-1, k-1, a.GetReCmplx(k-1, k-1))
					//                 (5) Swap row elements
					t = a.Get(k, k-1)
					a.Set(k, k-1, a.Get(kp-1, k-1))
					a.Set(kp-1, k-1, t)
				}
			} else {
				//              (*) Make sure that diagonal element of pivot is real
				a.Set(k-1, k-1, a.GetReCmplx(k-1, k-1))
				if kstep == 2 {
					a.Set(k, k, a.GetReCmplx(k, k))
				}
			}

			//           Update the trailing submatrix
			if kstep == 1 {
				//              1-by-1 pivot block D(k): column k of A now holds
				//
				//              W(k) = L(k)*D(k),
				//
				//              where L(k) is the k-th column of L
				if k < n {
					//                 Perform a rank-1 update of A(k+1:n,k+1:n) and
					//                 store L(k) in column k
					//
					//                 Handle division by a small number
					if math.Abs(a.GetRe(k-1, k-1)) >= sfmin {
						//                    Perform a rank-1 update of A(k+1:n,k+1:n) as
						//                    A := A - L(k)*D(k)*L(k)**T
						//                       = A - W(k)*(1/D(k))*W(k)**T
						d11 = one / a.GetRe(k-1, k-1)
						if err = a.Off(k, k).Her(uplo, n-k, -d11, a.Off(k, k-1).CVector(), 1); err != nil {
							panic(err)
						}

						//                    Store L(k) in column k
						a.Off(k, k-1).CVector().Dscal(n-k, d11, 1)
					} else {
						//                    Store L(k) in column k
						d11 = a.GetRe(k-1, k-1)
						for ii = k + 1; ii <= n; ii++ {
							a.Set(ii-1, k-1, a.Get(ii-1, k-1)/complex(d11, 0))
						}

						//                    Perform a rank-1 update of A(k+1:n,k+1:n) as
						//                    A := A - L(k)*D(k)*L(k)**T
						//                       = A - W(k)*(1/D(k))*W(k)**T
						//                       = A - (W(k)/D(k))*(D(k))*(W(k)/D(K))**T
						if err = a.Off(k, k).Her(uplo, n-k, -d11, a.Off(k, k-1).CVector(), 1); err != nil {
							panic(err)
						}
					}
				}

			} else {
				//              2-by-2 pivot block D(k): columns k and k+1 now hold
				//
				//              ( W(k) W(k+1) ) = ( L(k) L(k+1) )*D(k)
				//
				//              where L(k) and L(k+1) are the k-th and (k+1)-th columns
				//              of L
				//
				//
				//              Perform a rank-2 update of A(k+2:n,k+2:n) as
				//
				//              A := A - ( L(k) L(k+1) ) * D(k) * ( L(k) L(k+1) )**T
				//                 = A - ( ( A(k)A(k+1) )*inv(D(k) ) * ( A(k)A(k+1) )**T
				//
				//              and store L(k) and L(k+1) in columns k and k+1
				if k < n-1 {
					//                 D = |A21|
					d = Dlapy2(a.GetRe(k, k-1), a.GetIm(k, k-1))
					d11 = a.GetRe(k, k) / d
					d22 = a.GetRe(k-1, k-1) / d
					d21 = a.Get(k, k-1) / complex(d, 0)
					tt = one / (d11*d22 - one)

					for j = k + 2; j <= n; j++ {
						//                    Compute  D21 * ( W(k)W(k+1) ) * inv(D(k)) for row J
						wk = complex(tt, 0) * (complex(d11, 0)*a.Get(j-1, k-1) - d21*a.Get(j-1, k))
						wkp1 = complex(tt, 0) * (complex(d22, 0)*a.Get(j-1, k) - cmplx.Conj(d21)*a.Get(j-1, k-1))

						//                    Perform a rank-2 update of A(k+2:n,k+2:n)
						for i = j; i <= n; i++ {
							a.Set(i-1, j-1, a.Get(i-1, j-1)-(a.Get(i-1, k-1)/complex(d, 0))*cmplx.Conj(wk)-(a.Get(i-1, k)/complex(d, 0))*cmplx.Conj(wkp1))
						}

						//                    Store L(k) and L(k+1) in cols k and k+1 for row J
						a.Set(j-1, k-1, wk/complex(d, 0))
						a.Set(j-1, k, wkp1/complex(d, 0))
						//                    (*) Make sure that diagonal element of pivot is real
						a.Set(j-1, j-1, a.GetReCmplx(j-1, j-1))

					}

				}

			}

		}

		//        Store details of the interchanges in IPIV
		if kstep == 1 {
			(*ipiv)[k-1] = kp
		} else {
			(*ipiv)[k-1] = -p
			(*ipiv)[k] = -kp
		}

		//        Increase K and return to the start of the main loop
		k = k + kstep
		goto label40

	}
}
