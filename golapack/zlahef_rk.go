package golapack

import (
	"math"
	"math/cmplx"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/mat"
)

// ZlahefRk computes a partial factorization of a complex Hermitian
// matrix A using the bounded Bunch-Kaufman (rook) diagonal
// pivoting method. The partial factorization has the form:
//
// A  =  ( I  U12 ) ( A11  0  ) (  I       0    )  if UPLO = 'U', or:
//       ( 0  U22 ) (  0   D  ) ( U12**H U22**H )
//
// A  =  ( L11  0 ) (  D   0  ) ( L11**H L21**H )  if UPLO = 'L',
//       ( L21  I ) (  0  A22 ) (  0       I    )
//
// where the order of D is at most NB. The actual order is returned in
// the argument KB, and is either NB or NB-1, or N if N <= NB.
//
// ZLAHEF_RK is an auxiliary routine called by ZHETRF_RK. It uses
// blocked code (calling Level 3 BLAS) to update the submatrix
// A11 (if UPLO = 'U') or A22 (if UPLO = 'L').
func ZlahefRk(uplo mat.MatUplo, n, nb int, a *mat.CMatrix, e *mat.CVector, ipiv *[]int, w *mat.CMatrix) (kb, info int) {
	var done bool
	var cone, czero, d11, d21, d22 complex128
	var absakk, alpha, colmax, dtemp, eight, one, r1, rowmax, sevten, sfmin, t, zero float64
	var ii, imax, itemp, j, jb, jj, jmax, k, kk, kkw, kp, kstep, kw, p int
	var err error

	zero = 0.0
	one = 1.0
	cone = (1.0 + 0.0*1i)
	eight = 8.0
	sevten = 17.0
	czero = (0.0 + 0.0*1i)

	//     Initialize ALPHA for use in choosing pivot block size.
	alpha = (one + math.Sqrt(sevten)) / eight

	//     Compute machine safe minimum
	sfmin = Dlamch(SafeMinimum)

	if uplo == Upper {
		//        Factorize the trailing columns of A using the upper triangle
		//        of A and working backwards, and compute the matrix W = U12*D
		//        for use in updating A11 (note that conjg(W) is actually stored)
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
		if k > 1 {
			goblas.Zcopy(k-1, a.CVector(0, k-1, 1), w.CVector(0, kw-1, 1))
		}
		w.Set(k-1, kw-1, a.GetReCmplx(k-1, k-1))
		if k < n {
			if err = goblas.Zgemv(NoTrans, k, n-k, -cone, a.Off(0, k), w.CVector(k-1, kw), cone, w.CVector(0, kw-1, 1)); err != nil {
				panic(err)
			}
			w.Set(k-1, kw-1, w.GetReCmplx(k-1, kw-1))
		}

		//        Determine rows and columns to be interchanged and whether
		//        a 1-by-1 or 2-by-2 pivot block will be used
		absakk = math.Abs(w.GetRe(k-1, kw-1))

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
			a.Set(k-1, k-1, w.GetReCmplx(k-1, kw-1))
			if k > 1 {
				goblas.Zcopy(k-1, w.CVector(0, kw-1, 1), a.CVector(0, k-1, 1))
			}

			//           Set E( K ) to zero
			if k > 1 {
				e.Set(k-1, czero)
			}

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
				//              Lop until pivot found
				done = false

			label12:
				;

				//                 BEGIN pivot search loop body
				//
				//
				//                 Copy column IMAX to column KW-1 of W and update it
				if imax > 1 {
					goblas.Zcopy(imax-1, a.CVector(0, imax-1, 1), w.CVector(0, kw-1-1, 1))
				}
				w.Set(imax-1, kw-1-1, a.GetReCmplx(imax-1, imax-1))

				goblas.Zcopy(k-imax, a.CVector(imax-1, imax), w.CVector(imax, kw-1-1, 1))
				Zlacgv(k-imax, w.CVector(imax, kw-1-1, 1))

				if k < n {
					if err = goblas.Zgemv(NoTrans, k, n-k, -cone, a.Off(0, k), w.CVector(imax-1, kw), cone, w.CVector(0, kw-1-1, 1)); err != nil {
						panic(err)
					}
					w.Set(imax-1, kw-1-1, w.GetReCmplx(imax-1, kw-1-1))
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

				//                 Case(2)
				//                 Equivalent to testing for
				//                 ABS( REAL( W( IMAX,KW-1 ) ) ).GE.ALPHA*ROWMAX
				//                 (used to handle NaN and Inf)
				if !(math.Abs(w.GetRe(imax-1, kw-1-1)) < alpha*rowmax) {
					//                    interchange rows and columns K and IMAX,
					//                    use 1-by-1 pivot block
					kp = imax

					//                    copy column KW-1 of W to column KW of W
					goblas.Zcopy(k, w.CVector(0, kw-1-1, 1), w.CVector(0, kw-1, 1))

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

					//                    Copy updated JMAXth (next IMAXth) column to Kth of W
					goblas.Zcopy(k, w.CVector(0, kw-1-1, 1), w.CVector(0, kw-1, 1))

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

			//           KKW is the column of W which corresponds to column KK of A
			kkw = nb + kk - n

			//           Interchange rows and columns P and K.
			//           Updated column P is already stored in column KW of W.
			if (kstep == 2) && (p != k) {
				//              Copy non-updated column K to column P of submatrix A
				//              at step K. No need to copy element into columns
				//              K and K-1 of A for 2-by-2 pivot, since these columns
				//              will be later overwritten.
				a.Set(p-1, p-1, a.GetReCmplx(k-1, k-1))
				goblas.Zcopy(k-1-p, a.CVector(p, k-1, 1), a.CVector(p-1, p))
				Zlacgv(k-1-p, a.CVector(p-1, p))
				if p > 1 {
					goblas.Zcopy(p-1, a.CVector(0, k-1, 1), a.CVector(0, p-1, 1))
				}

				//              Interchange rows K and P in the last K+1 to N columns of A
				//              (columns K and K-1 of A for 2-by-2 pivot will be
				//              later overwritten). Interchange rows K and P
				//              in last KKW to NB columns of W.
				if k < n {
					goblas.Zswap(n-k, a.CVector(k-1, k), a.CVector(p-1, k))
				}
				goblas.Zswap(n-kk+1, w.CVector(k-1, kkw-1), w.CVector(p-1, kkw-1))
			}

			//           Interchange rows and columns KP and KK.
			//           Updated column KP is already stored in column KKW of W.
			if kp != kk {
				//              Copy non-updated column KK to column KP of submatrix A
				//              at step K. No need to copy element into column K
				//              (or K and K-1 for 2-by-2 pivot) of A, since these columns
				//              will be later overwritten.
				a.Set(kp-1, kp-1, a.GetReCmplx(kk-1, kk-1))
				goblas.Zcopy(kk-1-kp, a.CVector(kp, kk-1, 1), a.CVector(kp-1, kp))
				Zlacgv(kk-1-kp, a.CVector(kp-1, kp))
				if kp > 1 {
					goblas.Zcopy(kp-1, a.CVector(0, kk-1, 1), a.CVector(0, kp-1, 1))
				}

				//              Interchange rows KK and KP in last K+1 to N columns of A
				//              (columns K (or K and K-1 for 2-by-2 pivot) of A will be
				//              later overwritten). Interchange rows KK and KP
				//              in last KKW to NB columns of W.
				if k < n {
					goblas.Zswap(n-k, a.CVector(kk-1, k), a.CVector(kp-1, k))
				}
				goblas.Zswap(n-kk+1, w.CVector(kk-1, kkw-1), w.CVector(kp-1, kkw-1))
			}

			if kstep == 1 {
				//              1-by-1 pivot block D(k): column kw of W now holds
				//
				//              W(kw) = U(k)*D(k),
				//
				//              where U(k) is the k-th column of U
				//
				//              (1) Store subdiag. elements of column U(k)
				//              and 1-by-1 block D(k) in column k of A.
				//              (NOTE: Diagonal element U(k,k) is a UNIT element
				//              and not stored)
				//                 A(k,k) := D(k,k) = W(k,kw)
				//                 A(1:k-1,k) := U(1:k-1,k) = W(1:k-1,kw)/D(k,k)
				//
				//              (NOTE: No need to use for Hermitian matrix
				//              A( K, K ) = REAL( W( K, K) ) to separately copy diagonal
				//              element D(k,k) from W (potentially saves only one load))
				goblas.Zcopy(k, w.CVector(0, kw-1, 1), a.CVector(0, k-1, 1))
				if k > 1 {
					//                 (NOTE: No need to check if A(k,k) is NOT ZERO,
					//                  since that was ensured earlier in pivot search:
					//                  case A(k,k) = 0 falls into 2x2 pivot case(3))
					//
					//                 Handle division by a small number
					t = a.GetRe(k-1, k-1)
					if math.Abs(t) >= sfmin {
						r1 = one / t
						goblas.Zdscal(k-1, r1, a.CVector(0, k-1, 1))
					} else {
						for ii = 1; ii <= k-1; ii++ {
							a.Set(ii-1, k-1, a.Get(ii-1, k-1)/complex(t, 0))
						}
					}

					//                 (2) Conjugate column W(kw)
					Zlacgv(k-1, w.CVector(0, kw-1, 1))

					//                 Store the superdiagonal element of D in array E
					e.Set(k-1, czero)

				}

			} else {
				//              2-by-2 pivot block D(k): columns kw and kw-1 of W now hold
				//
				//              ( W(kw-1) W(kw) ) = ( U(k-1) U(k) )*D(k)
				//
				//              where U(k) and U(k-1) are the k-th and (k-1)-th columns
				//              of U
				//
				//              (1) Store U(1:k-2,k-1) and U(1:k-2,k) and 2-by-2
				//              block D(k-1:k,k-1:k) in columns k-1 and k of A.
				//              (NOTE: 2-by-2 diagonal block U(k-1:k,k-1:k) is a UNIT
				//              block and not stored)
				//                 A(k-1:k,k-1:k) := D(k-1:k,k-1:k) = W(k-1:k,kw-1:kw)
				//                 A(1:k-2,k-1:k) := U(1:k-2,k:k-1:k) =
				//                 = W(1:k-2,kw-1:kw) * ( D(k-1:k,k-1:k)**(-1) )
				if k > 2 {
					//                 Factor out the columns of the inverse of 2-by-2 pivot
					//                 block D, so that each column contains 1, to reduce the
					//                 number of FLOPS when we multiply panel
					//                 ( W(kw-1) W(kw) ) by this inverse, i.e. by D**(-1).
					//
					//                 D**(-1) = ( d11 cj(d21) )**(-1) =
					//                           ( d21    d22 )
					//
					//                 = 1/(d11*d22-|d21|**2) * ( ( d22) (-cj(d21) ) ) =
					//                                          ( (-d21) (     d11 ) )
					//
					//                 = 1/(|d21|**2) * 1/((d11/cj(d21))*(d22/d21)-1) *
					//
					//                   * ( d21*( d22/d21 ) cmplx.Conj(d21)*(           - 1 ) ) =
					//                     (     (      -1 )           ( d11/cmplx.Conj(d21) ) )
					//
					//                 = 1/(|d21|**2) * 1/(D22*D11-1) *
					//
					//                   * ( d21*( D11 ) cmplx.Conj(d21)*(  -1 ) ) =
					//                     (     (  -1 )           ( D22 ) )
					//
					//                 = (1/|d21|**2) * T * ( d21*( D11 ) cmplx.Conj(d21)*(  -1 ) ) =
					//                                      (     (  -1 )           ( D22 ) )
					//
					//                 = ( (T/cmplx.Conj(d21))*( D11 ) (T/d21)*(  -1 ) ) =
					//                   (               (  -1 )         ( D22 ) )
					//
					//                 Handle division by a small number. (NOTE: order of
					//                 operations is important)
					//
					//                 = ( T*(( D11 )/cmplx.Conj(D21)) T*((  -1 )/D21 ) )
					//                   (   ((  -1 )          )   (( D22 )     ) ),
					//
					//                 where D11 = d22/d21,
					//                       D22 = d11/cmplx.Conj(d21),
					//                       D21 = d21,
					//                       T = 1/(D22*D11-1).
					//
					//                 (NOTE: No need to check for division by ZERO,
					//                  since that was ensured earlier in pivot search:
					//                  (a) d21 != 0 in 2x2 pivot case(4),
					//                      since |d21| should be larger than |d11| and |d22|;
					//                  (b) (D22*D11 - 1) != 0, since from (a),
					//                      both |D11| < 1, |D22| < 1, hence |D22*D11| << 1.)
					d21 = w.Get(k-1-1, kw-1)
					d11 = w.Get(k-1, kw-1) / cmplx.Conj(d21)
					d22 = w.Get(k-1-1, kw-1-1) / d21
					t = one / (real(d11*d22) - one)

					//                 Update elements in columns A(k-1) and A(k) as
					//                 dot products of rows of ( W(kw-1) W(kw) ) and columns
					//                 of D**(-1)
					for j = 1; j <= k-2; j++ {
						a.Set(j-1, k-1-1, complex(t, 0)*((d11*w.Get(j-1, kw-1-1)-w.Get(j-1, kw-1))/d21))
						a.Set(j-1, k-1, complex(t, 0)*((d22*w.Get(j-1, kw-1)-w.Get(j-1, kw-1-1))/cmplx.Conj(d21)))
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

				//              (2) Conjugate columns W(kw) and W(kw-1)
				Zlacgv(k-1, w.CVector(0, kw-1, 1))
				Zlacgv(k-2, w.CVector(0, kw-1-1, 1))

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
		//        A11 := A11 - U12*D*U12**H = A11 - U12*W**H
		//
		//        computing blocks of NB columns at a time (note that conjg(W) is
		//        actually stored)
		for j = ((k-1)/nb)*nb + 1; j >= 1; j -= nb {
			jb = min(nb, k-j+1)

			//           Update the upper triangle of the diagonal block
			for jj = j; jj <= j+jb-1; jj++ {
				a.Set(jj-1, jj-1, a.GetReCmplx(jj-1, jj-1))
				if err = goblas.Zgemv(NoTrans, jj-j+1, n-k, -cone, a.Off(j-1, k), w.CVector(jj-1, kw), cone, a.CVector(j-1, jj-1, 1)); err != nil {
					panic(err)
				}
				a.Set(jj-1, jj-1, a.GetReCmplx(jj-1, jj-1))
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
		//        for use in updating A22 (note that conjg(W) is actually stored)
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

		//        Copy column K of A to column K of W and update column K of W
		w.Set(k-1, k-1, a.GetReCmplx(k-1, k-1))
		if k < n {
			goblas.Zcopy(n-k, a.CVector(k, k-1, 1), w.CVector(k, k-1, 1))
		}
		if k > 1 {
			if err = goblas.Zgemv(NoTrans, n-k+1, k-1, -cone, a.Off(k-1, 0), w.CVector(k-1, 0), cone, w.CVector(k-1, k-1, 1)); err != nil {
				panic(err)
			}
			w.Set(k-1, k-1, w.GetReCmplx(k-1, k-1))
		}

		//        Determine rows and columns to be interchanged and whether
		//        a 1-by-1 or 2-by-2 pivot block will be used
		absakk = math.Abs(w.GetRe(k-1, k-1))

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
			a.Set(k-1, k-1, w.GetReCmplx(k-1, k-1))
			if k < n {
				goblas.Zcopy(n-k, w.CVector(k, k-1, 1), a.CVector(k, k-1, 1))
			}

			//           Set E( K ) to zero
			if k < n {
				e.Set(k-1, czero)
			}

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
			label72:
				;

				//                 BEGIN pivot search loop body
				//
				//
				//                 Copy column IMAX to column k+1 of W and update it
				goblas.Zcopy(imax-k, a.CVector(imax-1, k-1), w.CVector(k-1, k, 1))
				Zlacgv(imax-k, w.CVector(k-1, k, 1))
				w.Set(imax-1, k, a.GetReCmplx(imax-1, imax-1))

				if imax < n {
					goblas.Zcopy(n-imax, a.CVector(imax, imax-1, 1), w.CVector(imax, k, 1))
				}

				if k > 1 {
					if err = goblas.Zgemv(NoTrans, n-k+1, k-1, -cone, a.Off(k-1, 0), w.CVector(imax-1, 0), cone, w.CVector(k-1, k, 1)); err != nil {
						panic(err)
					}
					w.Set(imax-1, k, w.GetReCmplx(imax-1, k))
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

				//                 Case(2)
				//                 Equivalent to testing for
				//                 ABS( REAL( W( IMAX,K+1 ) ) ).GE.ALPHA*ROWMAX
				//                 (used to handle NaN and Inf)
				if !(math.Abs(w.GetRe(imax-1, k)) < alpha*rowmax) {
					//                    interchange rows and columns K and IMAX,
					//                    use 1-by-1 pivot block
					kp = imax

					//                    copy column K+1 of W to column K of W
					goblas.Zcopy(n-k+1, w.CVector(k-1, k, 1), w.CVector(k-1, k-1, 1))

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

					//                    Copy updated JMAXth (next IMAXth) column to Kth of W
					goblas.Zcopy(n-k+1, w.CVector(k-1, k, 1), w.CVector(k-1, k-1, 1))

				}

				//                 End pivot search loop body
				if !done {
					goto label72
				}

			}

			//           END pivot search
			//
			//           ============================================================
			//
			//           KK is the column of A where pivoting step stopped
			kk = k + kstep - 1

			//           Interchange rows and columns P and K (only for 2-by-2 pivot).
			//           Updated column P is already stored in column K of W.
			if (kstep == 2) && (p != k) {
				//              Copy non-updated column KK-1 to column P of submatrix A
				//              at step K. No need to copy element into columns
				//              K and K+1 of A for 2-by-2 pivot, since these columns
				//              will be later overwritten.
				a.Set(p-1, p-1, a.GetReCmplx(k-1, k-1))
				goblas.Zcopy(p-k-1, a.CVector(k, k-1, 1), a.CVector(p-1, k))
				Zlacgv(p-k-1, a.CVector(p-1, k))
				if p < n {
					goblas.Zcopy(n-p, a.CVector(p, k-1, 1), a.CVector(p, p-1, 1))
				}

				//              Interchange rows K and P in first K-1 columns of A
				//              (columns K and K+1 of A for 2-by-2 pivot will be
				//              later overwritten). Interchange rows K and P
				//              in first KK columns of W.
				if k > 1 {
					goblas.Zswap(k-1, a.CVector(k-1, 0), a.CVector(p-1, 0))
				}
				goblas.Zswap(kk, w.CVector(k-1, 0), w.CVector(p-1, 0))
			}

			//           Interchange rows and columns KP and KK.
			//           Updated column KP is already stored in column KK of W.
			if kp != kk {
				//              Copy non-updated column KK to column KP of submatrix A
				//              at step K. No need to copy element into column K
				//              (or K and K+1 for 2-by-2 pivot) of A, since these columns
				//              will be later overwritten.
				a.Set(kp-1, kp-1, a.GetReCmplx(kk-1, kk-1))
				goblas.Zcopy(kp-kk-1, a.CVector(kk, kk-1, 1), a.CVector(kp-1, kk))
				Zlacgv(kp-kk-1, a.CVector(kp-1, kk))
				if kp < n {
					goblas.Zcopy(n-kp, a.CVector(kp, kk-1, 1), a.CVector(kp, kp-1, 1))
				}

				//              Interchange rows KK and KP in first K-1 columns of A
				//              (column K (or K and K+1 for 2-by-2 pivot) of A will be
				//              later overwritten). Interchange rows KK and KP
				//              in first KK columns of W.
				if k > 1 {
					goblas.Zswap(k-1, a.CVector(kk-1, 0), a.CVector(kp-1, 0))
				}
				goblas.Zswap(kk, w.CVector(kk-1, 0), w.CVector(kp-1, 0))
			}

			if kstep == 1 {
				//              1-by-1 pivot block D(k): column k of W now holds
				//
				//              W(k) = L(k)*D(k),
				//
				//              where L(k) is the k-th column of L
				//
				//              (1) Store subdiag. elements of column L(k)
				//              and 1-by-1 block D(k) in column k of A.
				//              (NOTE: Diagonal element L(k,k) is a UNIT element
				//              and not stored)
				//                 A(k,k) := D(k,k) = W(k,k)
				//                 A(k+1:N,k) := L(k+1:N,k) = W(k+1:N,k)/D(k,k)
				//
				//              (NOTE: No need to use for Hermitian matrix
				//              A( K, K ) = REAL( W( K, K) ) to separately copy diagonal
				//              element D(k,k) from W (potentially saves only one load))
				goblas.Zcopy(n-k+1, w.CVector(k-1, k-1, 1), a.CVector(k-1, k-1, 1))
				if k < n {
					//                 (NOTE: No need to check if A(k,k) is NOT ZERO,
					//                  since that was ensured earlier in pivot search:
					//                  case A(k,k) = 0 falls into 2x2 pivot case(3))
					//
					//                 Handle division by a small number
					t = a.GetRe(k-1, k-1)
					if math.Abs(t) >= sfmin {
						r1 = one / t
						goblas.Zdscal(n-k, r1, a.CVector(k, k-1, 1))
					} else {
						for ii = k + 1; ii <= n; ii++ {
							a.Set(ii-1, k-1, a.Get(ii-1, k-1)/complex(t, 0))
						}
					}

					//                 (2) Conjugate column W(k)
					Zlacgv(n-k, w.CVector(k, k-1, 1))

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
				//
				//              (1) Store L(k+2:N,k) and L(k+2:N,k+1) and 2-by-2
				//              block D(k:k+1,k:k+1) in columns k and k+1 of A.
				//              NOTE: 2-by-2 diagonal block L(k:k+1,k:k+1) is a UNIT
				//              block and not stored.
				//                 A(k:k+1,k:k+1) := D(k:k+1,k:k+1) = W(k:k+1,k:k+1)
				//                 A(k+2:N,k:k+1) := L(k+2:N,k:k+1) =
				//                 = W(k+2:N,k:k+1) * ( D(k:k+1,k:k+1)**(-1) )
				if k < n-1 {
					//
					//                 Factor out the columns of the inverse of 2-by-2 pivot
					//                 block D, so that each column contains 1, to reduce the
					//                 number of FLOPS when we multiply panel
					//                 ( W(kw-1) W(kw) ) by this inverse, i.e. by D**(-1).
					//
					//                 D**(-1) = ( d11 cj(d21) )**(-1) =
					//                           ( d21    d22 )
					//
					//                 = 1/(d11*d22-|d21|**2) * ( ( d22) (-cj(d21) ) ) =
					//                                          ( (-d21) (     d11 ) )
					//
					//                 = 1/(|d21|**2) * 1/((d11/cj(d21))*(d22/d21)-1) *
					//
					//                   * ( d21*( d22/d21 ) cmplx.Conj(d21)*(           - 1 ) ) =
					//                     (     (      -1 )           ( d11/cmplx.Conj(d21) ) )
					//
					//                 = 1/(|d21|**2) * 1/(D22*D11-1) *
					//
					//                   * ( d21*( D11 ) cmplx.Conj(d21)*(  -1 ) ) =
					//                     (     (  -1 )           ( D22 ) )
					//
					//                 = (1/|d21|**2) * T * ( d21*( D11 ) cmplx.Conj(d21)*(  -1 ) ) =
					//                                      (     (  -1 )           ( D22 ) )
					//
					//                 = ( (T/cmplx.Conj(d21))*( D11 ) (T/d21)*(  -1 ) ) =
					//                   (               (  -1 )         ( D22 ) )
					//
					//                 Handle division by a small number. (NOTE: order of
					//                 operations is important)
					//
					//                 = ( T*(( D11 )/cmplx.Conj(D21)) T*((  -1 )/D21 ) )
					//                   (   ((  -1 )          )   (( D22 )     ) ),
					//
					//                 where D11 = d22/d21,
					//                       D22 = d11/cmplx.Conj(d21),
					//                       D21 = d21,
					//                       T = 1/(D22*D11-1).
					//
					//                 (NOTE: No need to check for division by ZERO,
					//                  since that was ensured earlier in pivot search:
					//                  (a) d21 != 0 in 2x2 pivot case(4),
					//                      since |d21| should be larger than |d11| and |d22|;
					//                  (b) (D22*D11 - 1) != 0, since from (a),
					//                      both |D11| < 1, |D22| < 1, hence |D22*D11| << 1.)
					d21 = w.Get(k, k-1)
					d11 = w.Get(k, k) / d21
					d22 = w.Get(k-1, k-1) / cmplx.Conj(d21)
					t = one / (real(d11*d22) - one)

					//                 Update elements in columns A(k) and A(k+1) as
					//                 dot products of rows of ( W(k) W(k+1) ) and columns
					//                 of D**(-1)
					for j = k + 2; j <= n; j++ {
						a.Set(j-1, k-1, complex(t, 0)*((d11*w.Get(j-1, k-1)-w.Get(j-1, k))/cmplx.Conj(d21)))
						a.Set(j-1, k, complex(t, 0)*((d22*w.Get(j-1, k)-w.Get(j-1, k-1))/d21))
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

				//              (2) Conjugate columns W(k) and W(k+1)
				Zlacgv(n-k, w.CVector(k, k-1, 1))
				Zlacgv(n-k-1, w.CVector(k+2-1, k, 1))

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
		//        A22 := A22 - L21*D*L21**H = A22 - L21*W**H
		//
		//        computing blocks of NB columns at a time (note that conjg(W) is
		//        actually stored)
		for j = k; j <= n; j += nb {
			jb = min(nb, n-j+1)

			//           Update the lower triangle of the diagonal block
			for jj = j; jj <= j+jb-1; jj++ {
				a.Set(jj-1, jj-1, a.GetReCmplx(jj-1, jj-1))
				if err = goblas.Zgemv(NoTrans, j+jb-jj, k-1, -cone, a.Off(jj-1, 0), w.CVector(jj-1, 0), cone, a.CVector(jj-1, jj-1, 1)); err != nil {
					panic(err)
				}
				a.Set(jj-1, jj-1, a.GetReCmplx(jj-1, jj-1))
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
