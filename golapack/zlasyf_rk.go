package golapack

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/mat"
)

// ZlasyfRk computes a partial factorization of a complex symmetric
// matrix A using the bounded Bunch-Kaufman (rook) diagonal
// pivoting method. The partial factorization has the form:
//
// A  =  ( I  U12 ) ( A11  0  ) (  I       0    )  if UPLO = 'U', or:
//       ( 0  U22 ) (  0   D  ) ( U12**T U22**T )
//
// A  =  ( L11  0 ) (  D   0  ) ( L11**T L21**T )  if UPLO = 'L',
//       ( L21  I ) (  0  A22 ) (  0       I    )
//
// where the order of D is at most NB. The actual order is returned in
// the argument KB, and is either NB or NB-1, or N if N <= NB.
//
// ZLASYF_RK is an auxiliary routine called by ZSYTRF_RK. It uses
// blocked code (calling Level 3 BLAS) to update the submatrix
// A11 (if UPLO = 'U') or A22 (if UPLO = 'L').
func ZlasyfRk(uplo mat.MatUplo, n, nb int, a *mat.CMatrix, e *mat.CVector, ipiv *[]int, w *mat.CMatrix) (kb, info int) {
	var done bool
	var cone, czero, d11, d12, d21, d22, r1, t complex128
	var absakk, alpha, colmax, dtemp, eight, one, rowmax, sevten, sfmin, zero float64
	var ii, imax, itemp, j, jb, jj, jmax, k, kk, kkw, kp, kstep, kw, p int
	var err error

	zero = 0.0
	one = 1.0
	eight = 8.0
	sevten = 17.0
	cone = (1.0 + 0.0*1i)
	czero = (0.0 + 0.0*1i)

	//     Initialize ALPHA for use in choosing pivot block size.
	alpha = (one + math.Sqrt(sevten)) / eight

	//     Compute machine safe minimum
	sfmin = Dlamch(SafeMinimum)

	if uplo == Upper {
		//        Factorize the trailing columns of A using the upper triangle
		//        of A and working backwards, and compute the matrix W = U12*D
		//        for use in updating A11
		//
		//        Initialize the first entry of array E, where superdiagonal
		//        elements of D are stored
		e.Set(0, czero)

		//        K is the main loop index, decreasing from N in steps of 1 or 2
		k = n
	label10:
		;

		//        KW is the column of W which corresponds to column K of A
		kw = nb + k - n

		//        Exit from loop
		if (k <= n-nb+1 && nb < n) || k < 1 {
			goto label30
		}

		kstep = 1
		p = k

		//        Copy column K of A to column KW of W and update it
		goblas.Zcopy(k, a.CVector(0, k-1, 1), w.CVector(0, kw-1, 1))
		if k < n {
			if err = goblas.Zgemv(NoTrans, k, n-k, -cone, a.Off(0, k), w.CVector(k-1, kw), cone, w.CVector(0, kw-1, 1)); err != nil {
				panic(err)
			}
		}

		//        Determine rows and columns to be interchanged and whether
		//        a 1-by-1 or 2-by-2 pivot block will be used
		absakk = cabs1(w.Get(k-1, kw-1))

		//        IMAX is the row-index of the largest off-diagonal element in
		//        column K, and COLMAX is its absolute value.
		//        Determine both COLMAX and IMAX.
		if k > 1 {
			imax = goblas.Izamax(k-1, w.CVector(0, kw-1, 1))
			colmax = cabs1(w.Get(imax-1, kw-1))
		} else {
			colmax = zero
		}

		if math.Max(absakk, colmax) == zero {
			//           Column K is zero or underflow: set INFO and continue
			if info == 0 {
				info = k
			}
			kp = k
			goblas.Zcopy(k, w.CVector(0, kw-1, 1), a.CVector(0, k-1, 1))

			//           Set E( K ) to zero
			if k > 1 {
				e.Set(k-1, czero)
			}

		} else {
			//           ============================================================
			//
			//           Test for interchange
			//
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

				//                 Begin pivot search loop body
				//
				//
				//                 Copy column IMAX to column KW-1 of W and update it
				goblas.Zcopy(imax, a.CVector(0, imax-1, 1), w.CVector(0, kw-1-1, 1))
				goblas.Zcopy(k-imax, a.CVector(imax-1, imax), w.CVector(imax, kw-1-1, 1))

				if k < n {
					if err = goblas.Zgemv(NoTrans, k, n-k, -cone, a.Off(0, k), w.CVector(imax-1, kw), cone, w.CVector(0, kw-1-1, 1)); err != nil {
						panic(err)
					}
				}

				//                 JMAX is the column-index of the largest off-diagonal
				//                 element in row IMAX, and ROWMAX is its absolute value.
				//                 Determine both ROWMAX and JMAX.
				if imax != k {
					jmax = imax + goblas.Izamax(k-imax, w.CVector(imax, kw-1-1, 1))
					rowmax = cabs1(w.Get(jmax-1, kw-1-1))
				} else {
					rowmax = zero
				}

				if imax > 1 {
					itemp = goblas.Izamax(imax-1, w.CVector(0, kw-1-1, 1))
					dtemp = cabs1(w.Get(itemp-1, kw-1-1))
					if dtemp > rowmax {
						rowmax = dtemp
						jmax = itemp
					}
				}

				//                 Equivalent to testing for
				//                 CABS1( W( IMAX, KW-1 ) ).GE.ALPHA*ROWMAX
				//                 (used to handle NaN and Inf)
				if !(cabs1(w.Get(imax-1, kw-1-1)) < alpha*rowmax) {
					//                    interchange rows and columns K and IMAX,
					//                    use 1-by-1 pivot block
					kp = imax

					//                    copy column KW-1 of W to column KW of W
					goblas.Zcopy(k, w.CVector(0, kw-1-1, 1), w.CVector(0, kw-1, 1))

					done = true

					//                 Equivalent to testing for ROWMAX.EQ.COLMAX,
					//                 (used to handle NaN and Inf)
				} else if (p == jmax) || (rowmax <= colmax) {
					//                    interchange rows and columns K-1 and IMAX,
					//                    use 2-by-2 pivot block
					kp = imax
					kstep = 2
					done = true
				} else {
					//                    Pivot not found: set params and repeat
					p = imax
					colmax = rowmax
					imax = jmax

					//                    Copy updated JMAXth (next IMAXth) column to Kth of W
					goblas.Zcopy(k, w.CVector(0, kw-1-1, 1), w.CVector(0, kw-1, 1))

				}

				//                 End pivot search loop body
				if !done {
					goto label12
				}

			}

			//           ============================================================
			kk = k - kstep + 1

			//           KKW is the column of W which corresponds to column KK of A
			kkw = nb + kk - n

			if (kstep == 2) && (p != k) {
				//              Copy non-updated column K to column P
				goblas.Zcopy(k-p, a.CVector(p, k-1, 1), a.CVector(p-1, p))
				goblas.Zcopy(p, a.CVector(0, k-1, 1), a.CVector(0, p-1, 1))

				//              Interchange rows K and P in last N-K+1 columns of A
				//              and last N-K+2 columns of W
				goblas.Zswap(n-k+1, a.CVector(k-1, k-1), a.CVector(p-1, k-1))
				goblas.Zswap(n-kk+1, w.CVector(k-1, kkw-1), w.CVector(p-1, kkw-1))
			}

			//           Updated column KP is already stored in column KKW of W
			if kp != kk {
				//              Copy non-updated column KK to column KP
				a.Set(kp-1, k-1, a.Get(kk-1, k-1))
				goblas.Zcopy(k-1-kp, a.CVector(kp, kk-1, 1), a.CVector(kp-1, kp))
				goblas.Zcopy(kp, a.CVector(0, kk-1, 1), a.CVector(0, kp-1, 1))

				//              Interchange rows KK and KP in last N-KK+1 columns
				//              of A and W
				goblas.Zswap(n-kk+1, a.CVector(kk-1, kk-1), a.CVector(kp-1, kk-1))
				goblas.Zswap(n-kk+1, w.CVector(kk-1, kkw-1), w.CVector(kp-1, kkw-1))
			}

			if kstep == 1 {
				//              1-by-1 pivot block D(k): column KW of W now holds
				//
				//              W(k) = U(k)*D(k)
				//
				//              where U(k) is the k-th column of U
				//
				//              Store U(k) in column k of A
				goblas.Zcopy(k, w.CVector(0, kw-1, 1), a.CVector(0, k-1, 1))
				if k > 1 {
					if cabs1(a.Get(k-1, k-1)) >= sfmin {
						r1 = cone / a.Get(k-1, k-1)
						goblas.Zscal(k-1, r1, a.CVector(0, k-1, 1))
					} else if a.Get(k-1, k-1) != czero {
						for ii = 1; ii <= k-1; ii++ {
							a.Set(ii-1, k-1, a.Get(ii-1, k-1)/a.Get(k-1, k-1))
						}
					}

					//                 Store the superdiagonal element of D in array E
					e.Set(k-1, czero)

				}

			} else {
				//              2-by-2 pivot block D(k): columns KW and KW-1 of W now
				//              hold
				//
				//              ( W(k-1) W(k) ) = ( U(k-1) U(k) )*D(k)
				//
				//              where U(k) and U(k-1) are the k-th and (k-1)-th columns
				//              of U
				if k > 2 {
					//                 Store U(k) and U(k-1) in columns k and k-1 of A
					d12 = w.Get(k-1-1, kw-1)
					d11 = w.Get(k-1, kw-1) / d12
					d22 = w.Get(k-1-1, kw-1-1) / d12
					t = cone / (d11*d22 - cone)
					for j = 1; j <= k-2; j++ {
						a.Set(j-1, k-1-1, t*((d11*w.Get(j-1, kw-1-1)-w.Get(j-1, kw-1))/d12))
						a.Set(j-1, k-1, t*((d22*w.Get(j-1, kw-1)-w.Get(j-1, kw-1-1))/d12))
					}
				}

				//              Copy diagonal elements of D(K) to A,
				//              copy superdiagonal element of D(K) to E(K) and
				//              ZERO out superdiagonal entry of A
				a.Set(k-1-1, k-1-1, w.Get(k-1-1, kw-1-1))
				a.Set(k-1-1, k-1, czero)
				a.Set(k-1, k-1, w.Get(k-1, kw-1))
				e.Set(k-1, w.Get(k-1-1, kw-1))
				e.Set(k-1-1, czero)

			}

			//           End column K is nonsingular
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

	label30:
		;

		//        Update the upper triangle of A11 (= A(1:k,1:k)) as
		//
		//        A11 := A11 - U12*D*U12**T = A11 - U12*W**T
		//
		//        computing blocks of NB columns at a time
		for j = ((k-1)/nb)*nb + 1; j >= 1; j -= nb {
			jb = min(nb, k-j+1)

			//           Update the upper triangle of the diagonal block
			for jj = j; jj <= j+jb-1; jj++ {
				if err = goblas.Zgemv(NoTrans, jj-j+1, n-k, -cone, a.Off(j-1, k), w.CVector(jj-1, kw), cone, a.CVector(j-1, jj-1, 1)); err != nil {
					panic(err)
				}
			}

			//           Update the rectangular superdiagonal block
			if j >= 2 {
				if err = goblas.Zgemm(NoTrans, Trans, j-1, jb, n-k, -cone, a.Off(0, k), w.Off(j-1, kw), cone, a.Off(0, j-1)); err != nil {
					panic(err)
				}
			}
		}

		//        Set KB to the number of columns factorized
		kb = n - k

	} else {
		//        Factorize the leading columns of A using the lower triangle
		//        of A and working forwards, and compute the matrix W = L21*D
		//        for use in updating A22
		//
		//        Initialize the unused last entry of the subdiagonal array E.
		e.Set(n-1, czero)

		//        K is the main loop index, increasing from 1 in steps of 1 or 2
		k = 1
	label70:
		;

		//        Exit from loop
		if (k >= nb && nb < n) || k > n {
			goto label90
		}

		kstep = 1
		p = k

		//        Copy column K of A to column K of W and update it
		goblas.Zcopy(n-k+1, a.CVector(k-1, k-1, 1), w.CVector(k-1, k-1, 1))
		if k > 1 {
			if err = goblas.Zgemv(NoTrans, n-k+1, k-1, -cone, a.Off(k-1, 0), w.CVector(k-1, 0), cone, w.CVector(k-1, k-1, 1)); err != nil {
				panic(err)
			}
		}

		//        Determine rows and columns to be interchanged and whether
		//        a 1-by-1 or 2-by-2 pivot block will be used
		absakk = cabs1(w.Get(k-1, k-1))

		//        IMAX is the row-index of the largest off-diagonal element in
		//        column K, and COLMAX is its absolute value.
		//        Determine both COLMAX and IMAX.
		if k < n {
			imax = k + goblas.Izamax(n-k, w.CVector(k, k-1, 1))
			colmax = cabs1(w.Get(imax-1, k-1))
		} else {
			colmax = zero
		}

		if math.Max(absakk, colmax) == zero {
			//           Column K is zero or underflow: set INFO and continue
			if info == 0 {
				info = k
			}
			kp = k
			goblas.Zcopy(n-k+1, w.CVector(k-1, k-1, 1), a.CVector(k-1, k-1, 1))

			//           Set E( K ) to zero
			if k < n {
				e.Set(k-1, czero)
			}

		} else {
			//           ============================================================
			//
			//           Test for interchange
			//
			//           Equivalent to testing for ABSAKK.GE.ALPHA*COLMAX
			//           (used to handle NaN and Inf)
			if !(absakk < alpha*colmax) {
				//              no interchange, use 1-by-1 pivot block
				kp = k

			} else {

				done = false

				//              Loop until pivot found
			label72:
				;

				//                 Begin pivot search loop body
				//
				//
				//                 Copy column IMAX to column K+1 of W and update it
				goblas.Zcopy(imax-k, a.CVector(imax-1, k-1), w.CVector(k-1, k, 1))
				goblas.Zcopy(n-imax+1, a.CVector(imax-1, imax-1, 1), w.CVector(imax-1, k, 1))
				if k > 1 {
					if err = goblas.Zgemv(NoTrans, n-k+1, k-1, -cone, a.Off(k-1, 0), w.CVector(imax-1, 0), cone, w.CVector(k-1, k, 1)); err != nil {
						panic(err)
					}
				}

				//                 JMAX is the column-index of the largest off-diagonal
				//                 element in row IMAX, and ROWMAX is its absolute value.
				//                 Determine both ROWMAX and JMAX.
				if imax != k {
					jmax = k - 1 + goblas.Izamax(imax-k, w.CVector(k-1, k, 1))
					rowmax = cabs1(w.Get(jmax-1, k))
				} else {
					rowmax = zero
				}

				if imax < n {
					itemp = imax + goblas.Izamax(n-imax, w.CVector(imax, k, 1))
					dtemp = cabs1(w.Get(itemp-1, k))
					if dtemp > rowmax {
						rowmax = dtemp
						jmax = itemp
					}
				}

				//                 Equivalent to testing for
				//                 CABS1( W( IMAX, K+1 ) ).GE.ALPHA*ROWMAX
				//                 (used to handle NaN and Inf)
				if !(cabs1(w.Get(imax-1, k)) < alpha*rowmax) {
					//                    interchange rows and columns K and IMAX,
					//                    use 1-by-1 pivot block
					kp = imax

					//                    copy column K+1 of W to column K of W
					goblas.Zcopy(n-k+1, w.CVector(k-1, k, 1), w.CVector(k-1, k-1, 1))

					done = true

					//                 Equivalent to testing for ROWMAX.EQ.COLMAX,
					//                 (used to handle NaN and Inf)
				} else if (p == jmax) || (rowmax <= colmax) {
					//                    interchange rows and columns K+1 and IMAX,
					//                    use 2-by-2 pivot block
					kp = imax
					kstep = 2
					done = true
				} else {
					//                    Pivot not found: set params and repeat
					p = imax
					colmax = rowmax
					imax = jmax

					//                    Copy updated JMAXth (next IMAXth) column to Kth of W
					goblas.Zcopy(n-k+1, w.CVector(k-1, k, 1), w.CVector(k-1, k-1, 1))

				}

				//                 End pivot search loop body
				if !done {
					goto label72
				}

			}

			//           ============================================================
			kk = k + kstep - 1

			if (kstep == 2) && (p != k) {
				//              Copy non-updated column K to column P
				goblas.Zcopy(p-k, a.CVector(k-1, k-1, 1), a.CVector(p-1, k-1))
				goblas.Zcopy(n-p+1, a.CVector(p-1, k-1, 1), a.CVector(p-1, p-1, 1))

				//              Interchange rows K and P in first K columns of A
				//              and first K+1 columns of W
				goblas.Zswap(k, a.CVector(k-1, 0), a.CVector(p-1, 0))
				goblas.Zswap(kk, w.CVector(k-1, 0), w.CVector(p-1, 0))
			}

			//           Updated column KP is already stored in column KK of W
			if kp != kk {
				//              Copy non-updated column KK to column KP
				a.Set(kp-1, k-1, a.Get(kk-1, k-1))
				goblas.Zcopy(kp-k-1, a.CVector(k, kk-1, 1), a.CVector(kp-1, k))
				goblas.Zcopy(n-kp+1, a.CVector(kp-1, kk-1, 1), a.CVector(kp-1, kp-1, 1))

				//              Interchange rows KK and KP in first KK columns of A and W
				goblas.Zswap(kk, a.CVector(kk-1, 0), a.CVector(kp-1, 0))
				goblas.Zswap(kk, w.CVector(kk-1, 0), w.CVector(kp-1, 0))
			}

			if kstep == 1 {
				//              1-by-1 pivot block D(k): column k of W now holds
				//
				//              W(k) = L(k)*D(k)
				//
				//              where L(k) is the k-th column of L
				//
				//              Store L(k) in column k of A
				goblas.Zcopy(n-k+1, w.CVector(k-1, k-1, 1), a.CVector(k-1, k-1, 1))
				if k < n {
					if cabs1(a.Get(k-1, k-1)) >= sfmin {
						r1 = cone / a.Get(k-1, k-1)
						goblas.Zscal(n-k, r1, a.CVector(k, k-1, 1))
					} else if a.Get(k-1, k-1) != czero {
						for ii = k + 1; ii <= n; ii++ {
							a.Set(ii-1, k-1, a.Get(ii-1, k-1)/a.Get(k-1, k-1))
						}
					}

					//                 Store the subdiagonal element of D in array E
					e.Set(k-1, czero)

				}

			} else {
				//              2-by-2 pivot block D(k): columns k and k+1 of W now hold
				//
				//              ( W(k) W(k+1) ) = ( L(k) L(k+1) )*D(k)
				//
				//              where L(k) and L(k+1) are the k-th and (k+1)-th columns
				//              of L
				if k < n-1 {
					//                 Store L(k) and L(k+1) in columns k and k+1 of A
					d21 = w.Get(k, k-1)
					d11 = w.Get(k, k) / d21
					d22 = w.Get(k-1, k-1) / d21
					t = cone / (d11*d22 - cone)
					for j = k + 2; j <= n; j++ {
						a.Set(j-1, k-1, t*((d11*w.Get(j-1, k-1)-w.Get(j-1, k))/d21))
						a.Set(j-1, k, t*((d22*w.Get(j-1, k)-w.Get(j-1, k-1))/d21))
					}
				}

				//              Copy diagonal elements of D(K) to A,
				//              copy subdiagonal element of D(K) to E(K) and
				//              ZERO out subdiagonal entry of A
				a.Set(k-1, k-1, w.Get(k-1, k-1))
				a.Set(k, k-1, czero)
				a.Set(k, k, w.Get(k, k))
				e.Set(k-1, w.Get(k, k-1))
				e.Set(k, czero)

			}

			//           End column K is nonsingular
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
		goto label70

	label90:
		;

		//        Update the lower triangle of A22 (= A(k:n,k:n)) as
		//
		//        A22 := A22 - L21*D*L21**T = A22 - L21*W**T
		//
		//        computing blocks of NB columns at a time
		for j = k; j <= n; j += nb {
			jb = min(nb, n-j+1)

			//           Update the lower triangle of the diagonal block
			for jj = j; jj <= j+jb-1; jj++ {
				if err = goblas.Zgemv(NoTrans, j+jb-jj, k-1, -cone, a.Off(jj-1, 0), w.CVector(jj-1, 0), cone, a.CVector(jj-1, jj-1, 1)); err != nil {
					panic(err)
				}
			}

			//           Update the rectangular subdiagonal block
			if j+jb <= n {
				if err = goblas.Zgemm(NoTrans, Trans, n-j-jb+1, jb, k-1, -cone, a.Off(j+jb-1, 0), w.Off(j-1, 0), cone, a.Off(j+jb-1, j-1)); err != nil {
					panic(err)
				}
			}
		}

		//        Set KB to the number of columns factorized
		kb = k - 1

	}

	return
}
