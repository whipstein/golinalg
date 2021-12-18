package golapack

import (
	"math"
	"math/cmplx"

	"github.com/whipstein/golinalg/mat"
)

// Zlahef computes a partial factorization of a complex Hermitian
// matrix A using the Bunch-Kaufman diagonal pivoting method. The
// partial factorization has the form:
//
// A  =  ( I  U12 ) ( A11  0  ) (  I      0     )  if UPLO = 'U', or:
//       ( 0  U22 ) (  0   D  ) ( U12**H U22**H )
//
// A  =  ( L11  0 ) (  D   0  ) ( L11**H L21**H )  if UPLO = 'L'
//       ( L21  I ) (  0  A22 ) (  0      I     )
//
// where the order of D is at most NB. The actual order is returned in
// the argument KB, and is either NB or NB-1, or N if N <= NB.
// Note that U**H denotes the conjugate transpose of U.
//
// ZLAHEF is an auxiliary routine called by ZHETRF. It uses blocked code
// (calling Level 3 BLAS) to update the submatrix A11 (if UPLO = 'U') or
// A22 (if UPLO = 'L').
func Zlahef(uplo mat.MatUplo, n, nb int, a *mat.CMatrix, ipiv *[]int, w *mat.CMatrix) (kb, info int) {
	var cone, d11, d21, d22 complex128
	var absakk, alpha, colmax, eight, one, r1, rowmax, sevten, t, zero float64
	var imax, j, jb, jj, jmax, jp, k, kk, kkw, kp, kstep, kw int
	var err error

	zero = 0.0
	one = 1.0
	cone = (1.0 + 0.0*1i)
	eight = 8.0
	sevten = 17.0

	//     Initialize ALPHA for use in choosing pivot block size.
	alpha = (one + math.Sqrt(sevten)) / eight

	if uplo == Upper {
		//        Factorize the trailing columns of A using the upper triangle
		//        of A and working backwards, and compute the matrix W = U12*D
		//        for use in updating A11 (note that conjg(W) is actually stored)
		//
		//        K is the main loop index, decreasing from N in steps of 1 or 2
		//
		//        KW is the column of W which corresponds to column K of A
		k = n
	label10:
		;
		kw = nb + k - n

		//        Exit from loop
		if (k <= n-nb+1 && nb < n) || k < 1 {
			goto label30
		}

		kstep = 1

		//        Copy column K of A to column KW of W and update it
		w.Off(0, kw-1).CVector().Copy(k-1, a.Off(0, k-1).CVector(), 1, 1)
		w.Set(k-1, kw-1, a.GetReCmplx(k-1, k-1))
		if k < n {
			err = w.Off(0, kw-1).CVector().Gemv(NoTrans, k, n-k, -cone, a.Off(0, k), w.Off(k-1, kw).CVector(), w.Rows, cone, 1)
			w.Set(k-1, kw-1, w.GetReCmplx(k-1, kw-1))
		}

		//        Determine rows and columns to be interchanged and whether
		//        a 1-by-1 or 2-by-2 pivot block will be used
		absakk = math.Abs(w.GetRe(k-1, kw-1))

		//        IMAX is the row-index of the largest off-diagonal element in
		//        column K, and COLMAX is its absolute value.
		//        Determine both COLMAX and IMAX.
		if k > 1 {
			imax = w.Off(0, kw-1).CVector().Iamax(k-1, 1)
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
			a.Set(k-1, k-1, a.GetReCmplx(k-1, k-1))
		} else {
			//           ============================================================
			//
			//           BEGIN pivot search
			//
			//           Case(1)
			if absakk >= alpha*colmax {
				//              no interchange, use 1-by-1 pivot block
				kp = k
			} else {
				//              BEGIN pivot search along IMAX row
				//
				//
				//              Copy column IMAX to column KW-1 of W and update it
				w.Off(0, kw-1-1).CVector().Copy(imax-1, a.Off(0, imax-1).CVector(), 1, 1)
				w.Set(imax-1, kw-1-1, a.GetReCmplx(imax-1, imax-1))
				w.Off(imax, kw-1-1).CVector().Copy(k-imax, a.Off(imax-1, imax).CVector(), a.Rows, 1)
				Zlacgv(k-imax, w.Off(imax, kw-1-1).CVector(), 1)
				if k < n {
					if err = w.Off(0, kw-1-1).CVector().Gemv(NoTrans, k, n-k, -cone, a.Off(0, k), w.Off(imax-1, kw).CVector(), w.Rows, cone, 1); err != nil {
						panic(err)
					}
					w.Set(imax-1, kw-1-1, w.GetReCmplx(imax-1, kw-1-1))
				}

				//              JMAX is the column-index of the largest off-diagonal
				//              element in row IMAX, and ROWMAX is its absolute value.
				//              Determine only ROWMAX.
				jmax = imax + w.Off(imax, kw-1-1).CVector().Iamax(k-imax, 1)
				rowmax = cabs1(w.Get(jmax-1, kw-1-1))
				if imax > 1 {
					jmax = w.Off(0, kw-1-1).CVector().Iamax(imax-1, 1)
					rowmax = math.Max(rowmax, cabs1(w.Get(jmax-1, kw-1-1)))
				}

				//              Case(2)
				if absakk >= alpha*colmax*(colmax/rowmax) {
					//                 no interchange, use 1-by-1 pivot block
					kp = k

					//              Case(3)
				} else if math.Abs(real(w.Get(imax-1, kw-1-1))) >= alpha*rowmax {
					//                 interchange rows and columns K and IMAX, use 1-by-1
					//                 pivot block
					kp = imax

					//                 copy column KW-1 of W to column KW of W
					w.Off(0, kw-1).CVector().Copy(k, w.Off(0, kw-1-1).CVector(), 1, 1)

					//              Case(4)
				} else {
					//                 interchange rows and columns K-1 and IMAX, use 2-by-2
					//                 pivot block
					kp = imax
					kstep = 2
				}

				//              END pivot search along IMAX row
			}

			//           END pivot search
			//
			//           ============================================================
			//
			//           KK is the column of A where pivoting step stopped
			kk = k - kstep + 1

			//           KKW is the column of W which corresponds to column KK of A
			kkw = nb + kk - n

			//           Interchange rows and columns KP and KK.
			//           Updated column KP is already stored in column KKW of W.
			if kp != kk {
				//              Copy non-updated column KK to column KP of submatrix A
				//              at step K. No need to copy element into column K
				//              (or K and K-1 for 2-by-2 pivot) of A, since these columns
				//              will be later overwritten.
				a.Set(kp-1, kp-1, a.GetReCmplx(kk-1, kk-1))
				a.Off(kp-1, kp).CVector().Copy(kk-1-kp, a.Off(kp, kk-1).CVector(), 1, a.Rows)
				Zlacgv(kk-1-kp, a.Off(kp-1, kp).CVector(), a.Rows)
				if kp > 1 {
					a.Off(0, kp-1).CVector().Copy(kp-1, a.Off(0, kk-1).CVector(), 1, 1)
				}

				//              Interchange rows KK and KP in last K+1 to N columns of A
				//              (columns K (or K and K-1 for 2-by-2 pivot) of A will be
				//              later overwritten). Interchange rows KK and KP
				//              in last KKW to NB columns of W.
				if k < n {
					a.Off(kp-1, k).CVector().Swap(n-k, a.Off(kk-1, k).CVector(), a.Rows, a.Rows)
				}
				w.Off(kp-1, kkw-1).CVector().Swap(n-kk+1, w.Off(kk-1, kkw-1).CVector(), w.Rows, w.Rows)
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
				//              A( K, K ) = DBLE( W( K, K) ) to separately copy diagonal
				//              element D(k,k) from W (potentially saves only one load))
				a.Off(0, k-1).CVector().Copy(k, w.Off(0, kw-1).CVector(), 1, 1)
				if k > 1 {
					//                 (NOTE: No need to check if A(k,k) is NOT ZERO,
					//                  since that was ensured earlier in pivot search:
					//                  case A(k,k) = 0 falls into 2x2 pivot case(4))
					r1 = one / real(a.Get(k-1, k-1))
					a.Off(0, k-1).CVector().Dscal(k-1, r1, 1)

					//                 (2) Conjugate column W(kw)
					Zlacgv(k-1, w.Off(0, kw-1).CVector(), 1)
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
					//                 = ( cmplx.Conj(D21)*( D11 ) D21*(  -1 ) )
					//                   (           (  -1 )     ( D22 ) ),
					//
					//                 where D11 = d22/d21,
					//                       D22 = d11/cmplx.Conj(d21),
					//                       D21 = T/d21,
					//                       T = 1/(D22*D11-1).
					//
					//                 (NOTE: No need to check for division by ZERO,
					//                  since that was ensured earlier in pivot search:
					//                  (a) d21 != 0, since in 2x2 pivot case(4)
					//                      |d21| should be larger than |d11| and |d22|;
					//                  (b) (D22*D11 - 1) != 0, since from (a),
					//                      both |D11| < 1, |D22| < 1, hence |D22*D11| << 1.)
					d21 = w.Get(k-1-1, kw-1)
					d11 = w.Get(k-1, kw-1) / cmplx.Conj(d21)
					d22 = w.Get(k-1-1, kw-1-1) / d21
					t = one / (real(d11*d22) - one)
					d21 = complex(t, 0) / d21

					//                 Update elements in columns A(k-1) and A(k) as
					//                 dot products of rows of ( W(kw-1) W(kw) ) and columns
					//                 of D**(-1)
					for j = 1; j <= k-2; j++ {
						a.Set(j-1, k-1-1, d21*(d11*w.Get(j-1, kw-1-1)-w.Get(j-1, kw-1)))
						a.Set(j-1, k-1, cmplx.Conj(d21)*(d22*w.Get(j-1, kw-1)-w.Get(j-1, kw-1-1)))
					}
				}

				//              Copy D(k) to A
				a.Set(k-1-1, k-1-1, w.Get(k-1-1, kw-1-1))
				a.Set(k-1-1, k-1, w.Get(k-1-1, kw-1))
				a.Set(k-1, k-1, w.Get(k-1, kw-1))

				//              (2) Conjugate columns W(kw) and W(kw-1)
				Zlacgv(k-1, w.Off(0, kw-1).CVector(), 1)
				Zlacgv(k-2, w.Off(0, kw-1-1).CVector(), 1)

			}

		}

		//        Store details of the interchanges in IPIV
		if kstep == 1 {
			(*ipiv)[k-1] = kp
		} else {
			(*ipiv)[k-1] = -kp
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
				if err = a.Off(j-1, jj-1).CVector().Gemv(NoTrans, jj-j+1, n-k, -cone, a.Off(j-1, k), w.Off(jj-1, kw).CVector(), w.Rows, cone, 1); err != nil {
					panic(err)
				}
				a.Set(jj-1, jj-1, a.GetReCmplx(jj-1, jj-1))
			}

			//           Update the rectangular superdiagonal block
			if err = a.Off(0, j-1).Gemm(NoTrans, Trans, j-1, jb, n-k, -cone, a.Off(0, k), w.Off(j-1, kw), cone); err != nil {
				panic(err)
			}
		}

		//        Put U12 in standard form by partially undoing the interchanges
		//        in columns k+1:n looping backwards from k+1 to n
		j = k + 1
	label60:
		;

		//           Undo the interchanges (if any) of rows JJ and JP at each
		//           step J
		//
		//           (Here, J is a diagonal index)
		jj = j
		jp = (*ipiv)[j-1]
		if jp < 0 {
			jp = -jp
			//              (Here, J is a diagonal index)
			j = j + 1
		}
		//           (NOTE: Here, J is used to determine row length. Length N-J+1
		//           of the rows to swap back doesn't include diagonal element)
		j = j + 1
		if jp != jj && j <= n {
			a.Off(jj-1, j-1).CVector().Swap(n-j+1, a.Off(jp-1, j-1).CVector(), a.Rows, a.Rows)
		}
		if j < n {
			goto label60
		}

		//        Set KB to the number of columns factorized
		kb = n - k

	} else {
		//        Factorize the leading columns of A using the lower triangle
		//        of A and working forwards, and compute the matrix W = L21*D
		//        for use in updating A22 (note that conjg(W) is actually stored)
		//
		//        K is the main loop index, increasing from 1 in steps of 1 or 2
		k = 1
	label70:
		;

		//        Exit from loop
		if (k >= nb && nb < n) || k > n {
			goto label90
		}

		kstep = 1

		//        Copy column K of A to column K of W and update it
		w.Set(k-1, k-1, a.GetReCmplx(k-1, k-1))
		if k < n {
			w.Off(k, k-1).CVector().Copy(n-k, a.Off(k, k-1).CVector(), 1, 1)
		}
		if err = w.Off(k-1, k-1).CVector().Gemv(NoTrans, n-k+1, k-1, -cone, a.Off(k-1, 0), w.Off(k-1, 0).CVector(), w.Rows, cone, 1); err != nil {
			panic(err)
		}
		w.Set(k-1, k-1, w.GetReCmplx(k-1, k-1))

		//        Determine rows and columns to be interchanged and whether
		//        a 1-by-1 or 2-by-2 pivot block will be used
		absakk = math.Abs(w.GetRe(k-1, k-1))

		//        IMAX is the row-index of the largest off-diagonal element in
		//        column K, and COLMAX is its absolute value.
		//        Determine both COLMAX and IMAX.
		if k < n {
			imax = k + w.Off(k, k-1).CVector().Iamax(n-k, 1)
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
			a.Set(k-1, k-1, a.GetReCmplx(k-1, k-1))
		} else {
			//           ============================================================
			//
			//           BEGIN pivot search
			//
			//           Case(1)
			if absakk >= alpha*colmax {
				//              no interchange, use 1-by-1 pivot block
				kp = k
			} else {
				//              BEGIN pivot search along IMAX row
				//
				//
				//              Copy column IMAX to column K+1 of W and update it
				w.Off(k-1, k).CVector().Copy(imax-k, a.Off(imax-1, k-1).CVector(), a.Rows, 1)
				Zlacgv(imax-k, w.Off(k-1, k).CVector(), 1)
				w.Set(imax-1, k, a.GetReCmplx(imax-1, imax-1))
				if imax < n {
					w.Off(imax, k).CVector().Copy(n-imax, a.Off(imax, imax-1).CVector(), 1, 1)
				}
				if err = w.Off(k-1, k).CVector().Gemv(NoTrans, n-k+1, k-1, -cone, a.Off(k-1, 0), w.Off(imax-1, 0).CVector(), w.Rows, cone, 1); err != nil {
					panic(err)
				}
				w.Set(imax-1, k, w.GetReCmplx(imax-1, k))

				//              JMAX is the column-index of the largest off-diagonal
				//              element in row IMAX, and ROWMAX is its absolute value.
				//              Determine only ROWMAX.
				jmax = k - 1 + w.Off(k-1, k).CVector().Iamax(imax-k, 1)
				rowmax = cabs1(w.Get(jmax-1, k))
				if imax < n {
					jmax = imax + w.Off(imax, k).CVector().Iamax(n-imax, 1)
					rowmax = math.Max(rowmax, cabs1(w.Get(jmax-1, k)))
				}

				//              Case(2)
				if absakk >= alpha*colmax*(colmax/rowmax) {
					//                 no interchange, use 1-by-1 pivot block
					kp = k

					//              Case(3)
				} else if math.Abs(real(w.Get(imax-1, k))) >= alpha*rowmax {
					//                 interchange rows and columns K and IMAX, use 1-by-1
					//                 pivot block
					kp = imax

					//                 copy column K+1 of W to column K of W
					w.Off(k-1, k-1).CVector().Copy(n-k+1, w.Off(k-1, k).CVector(), 1, 1)

					//              Case(4)
				} else {
					//                 interchange rows and columns K+1 and IMAX, use 2-by-2
					//                 pivot block
					kp = imax
					kstep = 2
				}

				//              END pivot search along IMAX row
			}

			//           END pivot search
			//
			//           ============================================================
			//
			//           KK is the column of A where pivoting step stopped
			kk = k + kstep - 1

			//           Interchange rows and columns KP and KK.
			//           Updated column KP is already stored in column KK of W.
			if kp != kk {
				//              Copy non-updated column KK to column KP of submatrix A
				//              at step K. No need to copy element into column K
				//              (or K and K+1 for 2-by-2 pivot) of A, since these columns
				//              will be later overwritten.
				a.Set(kp-1, kp-1, a.GetReCmplx(kk-1, kk-1))
				a.Off(kp-1, kk).CVector().Copy(kp-kk-1, a.Off(kk, kk-1).CVector(), 1, a.Rows)
				Zlacgv(kp-kk-1, a.Off(kp-1, kk).CVector(), a.Rows)
				if kp < n {
					a.Off(kp, kp-1).CVector().Copy(n-kp, a.Off(kp, kk-1).CVector(), 1, 1)
				}

				//              Interchange rows KK and KP in first K-1 columns of A
				//              (columns K (or K and K+1 for 2-by-2 pivot) of A will be
				//              later overwritten). Interchange rows KK and KP
				//              in first KK columns of W.
				if k > 1 {
					a.Off(kp-1, 0).CVector().Swap(k-1, a.Off(kk-1, 0).CVector(), a.Rows, a.Rows)
				}
				w.Off(kp-1, 0).CVector().Swap(kk, w.Off(kk-1, 0).CVector(), w.Rows, w.Rows)
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
				//              A( K, K ) = DBLE( W( K, K) ) to separately copy diagonal
				//              element D(k,k) from W (potentially saves only one load))
				a.Off(k-1, k-1).CVector().Copy(n-k+1, w.Off(k-1, k-1).CVector(), 1, 1)
				if k < n {
					//                 (NOTE: No need to check if A(k,k) is NOT ZERO,
					//                  since that was ensured earlier in pivot search:
					//                  case A(k,k) = 0 falls into 2x2 pivot case(4))
					r1 = one / real(a.Get(k-1, k-1))
					a.Off(k, k-1).CVector().Dscal(n-k, r1, 1)

					//                 (2) Conjugate column W(k)
					Zlacgv(n-k, w.Off(k, k-1).CVector(), 1)
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
				//              (NOTE: 2-by-2 diagonal block L(k:k+1,k:k+1) is a UNIT
				//              block and not stored)
				//                 A(k:k+1,k:k+1) := D(k:k+1,k:k+1) = W(k:k+1,k:k+1)
				//                 A(k+2:N,k:k+1) := L(k+2:N,k:k+1) =
				//                 = W(k+2:N,k:k+1) * ( D(k:k+1,k:k+1)**(-1) )
				if k < n-1 {
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
					//                 = ( cmplx.Conj(D21)*( D11 ) D21*(  -1 ) )
					//                   (           (  -1 )     ( D22 ) ),
					//
					//                 where D11 = d22/d21,
					//                       D22 = d11/cmplx.Conj(d21),
					//                       D21 = T/d21,
					//                       T = 1/(D22*D11-1).
					//
					//                 (NOTE: No need to check for division by ZERO,
					//                  since that was ensured earlier in pivot search:
					//                  (a) d21 != 0, since in 2x2 pivot case(4)
					//                      |d21| should be larger than |d11| and |d22|;
					//                  (b) (D22*D11 - 1) != 0, since from (a),
					//                      both |D11| < 1, |D22| < 1, hence |D22*D11| << 1.)
					d21 = w.Get(k, k-1)
					d11 = w.Get(k, k) / d21
					d22 = w.Get(k-1, k-1) / cmplx.Conj(d21)
					t = one / (real(d11*d22) - one)
					d21 = complex(t, 0) / d21

					//                 Update elements in columns A(k) and A(k+1) as
					//                 dot products of rows of ( W(k) W(k+1) ) and columns
					//                 of D**(-1)
					for j = k + 2; j <= n; j++ {
						a.Set(j-1, k-1, cmplx.Conj(d21)*(d11*w.Get(j-1, k-1)-w.Get(j-1, k)))
						a.Set(j-1, k, d21*(d22*w.Get(j-1, k)-w.Get(j-1, k-1)))
					}
				}

				//              Copy D(k) to A
				a.Set(k-1, k-1, w.Get(k-1, k-1))
				a.Set(k, k-1, w.Get(k, k-1))
				a.Set(k, k, w.Get(k, k))

				//              (2) Conjugate columns W(k) and W(k+1)
				Zlacgv(n-k, w.Off(k, k-1).CVector(), 1)
				Zlacgv(n-k-1, w.Off(k+2-1, k).CVector(), 1)

			}

		}

		//        Store details of the interchanges in IPIV
		if kstep == 1 {
			(*ipiv)[k-1] = kp
		} else {
			(*ipiv)[k-1] = -kp
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
				if err = a.Off(jj-1, jj-1).CVector().Gemv(NoTrans, j+jb-jj, k-1, -cone, a.Off(jj-1, 0), w.Off(jj-1, 0).CVector(), w.Rows, cone, 1); err != nil {
					panic(err)
				}
				a.Set(jj-1, jj-1, a.GetReCmplx(jj-1, jj-1))
			}

			//           Update the rectangular subdiagonal block
			if j+jb <= n {
				if err = a.Off(j+jb-1, j-1).Gemm(NoTrans, Trans, n-j-jb+1, jb, k-1, -cone, a.Off(j+jb-1, 0), w.Off(j-1, 0), cone); err != nil {
					panic(err)
				}
			}
		}

		//        Put L21 in standard form by partially undoing the interchanges
		//        of rows in columns 1:k-1 looping backwards from k-1 to 1
		j = k - 1
	label120:
		;

		//           Undo the interchanges (if any) of rows JJ and JP at each
		//           step J
		//
		//           (Here, J is a diagonal index)
		jj = j
		jp = (*ipiv)[j-1]
		if jp < 0 {
			jp = -jp
			//              (Here, J is a diagonal index)
			j = j - 1
		}
		//           (NOTE: Here, J is used to determine row length. Length J
		//           of the rows to swap back doesn't include diagonal element)
		j = j - 1
		if jp != jj && j >= 1 {
			a.Off(jj-1, 0).CVector().Swap(j, a.Off(jp-1, 0).CVector(), a.Rows, a.Rows)
		}
		if j > 1 {
			goto label120
		}

		//        Set KB to the number of columns factorized
		kb = k - 1

	}

	return
}
