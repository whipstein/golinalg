package golapack

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/mat"
)

// DlasyfRook computes a partial factorization of a real symmetric
// matrix A using the bounded Bunch-Kaufman ("rook") diagonal
// pivoting method. The partial factorization has the form:
//
// A  =  ( I  U12 ) ( A11  0  ) (  I       0    )  if UPLO = 'U', or:
//       ( 0  U22 ) (  0   D  ) ( U12**T U22**T )
//
// A  =  ( L11  0 ) (  D   0  ) ( L11**T L21**T )  if UPLO = 'L'
//       ( L21  I ) (  0  A22 ) (  0       I    )
//
// where the order of D is at most NB. The actual order is returned in
// the argument KB, and is either NB or NB-1, or N if N <= NB.
//
// DLASYF_ROOK is an auxiliary routine called by DSYTRF_ROOK. It uses
// blocked code (calling Level 3 BLAS) to update the submatrix
// A11 (if UPLO = 'U') or A22 (if UPLO = 'L').
func DlasyfRook(uplo byte, n, nb, kb *int, a *mat.Matrix, lda *int, ipiv *[]int, w *mat.Matrix, ldw *int, info *int) {
	var done bool
	var absakk, alpha, colmax, d11, d12, d21, d22, dtemp, eight, one, r1, rowmax, sevten, sfmin, t, zero float64
	var ii, imax, itemp, j, jb, jj, jmax, jp1, jp2, k, kk, kkw, kp, kstep, kw, p int
	var err error
	_ = err

	zero = 0.0
	one = 1.0
	eight = 8.0
	sevten = 17.0

	(*info) = 0

	//     Initialize ALPHA for use in choosing pivot block size.
	alpha = (one + math.Sqrt(sevten)) / eight

	//     Compute machine safe minimum
	sfmin = Dlamch(SafeMinimum)

	if uplo == 'U' {
		//        Factorize the trailing columns of A using the upper triangle
		//        of A and working backwards, and compute the matrix W = U12*D
		//        for use in updating A11
		//
		//        K is the main loop index, decreasing from N in steps of 1 or 2
		k = (*n)
	label10:
		;

		//        KW is the column of W which corresponds to column K of A
		kw = (*nb) + k - (*n)

		//        Exit from loop
		if (k <= (*n)-(*nb)+1 && (*nb) < (*n)) || k < 1 {
			goto label30
		}

		kstep = 1
		p = k

		//        Copy column K of A to column KW of W and update it
		goblas.Dcopy(k, a.Vector(0, k-1), 1, w.Vector(0, kw-1), 1)
		if k < (*n) {
			err = goblas.Dgemv(NoTrans, k, (*n)-k, -one, a.Off(0, k+1-1), *lda, w.Vector(k-1, kw+1-1), *ldw, one, w.Vector(0, kw-1), 1)
		}

		//        Determine rows and columns to be interchanged and whether
		//        a 1-by-1 or 2-by-2 pivot block will be used
		absakk = math.Abs(w.Get(k-1, kw-1))

		//        IMAX is the row-index of the largest off-diagonal element in
		//        column K, and COLMAX is its absolute value.
		//        Determine both COLMAX and IMAX.
		if k > 1 {
			imax = goblas.Idamax(k-1, w.Vector(0, kw-1), 1)
			colmax = math.Abs(w.Get(imax-1, kw-1))
		} else {
			colmax = zero
		}

		if maxf64(absakk, colmax) == zero {
			//           Column K is zero or underflow: set INFO and continue
			if (*info) == 0 {
				(*info) = k
			}
			kp = k
			goblas.Dcopy(k, w.Vector(0, kw-1), 1, a.Vector(0, k-1), 1)
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
				goblas.Dcopy(imax, a.Vector(0, imax-1), 1, w.Vector(0, kw-1-1), 1)
				goblas.Dcopy(k-imax, a.Vector(imax-1, imax+1-1), *lda, w.Vector(imax+1-1, kw-1-1), 1)

				if k < (*n) {
					err = goblas.Dgemv(NoTrans, k, (*n)-k, -one, a.Off(0, k+1-1), *lda, w.Vector(imax-1, kw+1-1), *ldw, one, w.Vector(0, kw-1-1), 1)
				}

				//                 JMAX is the column-index of the largest off-diagonal
				//                 element in row IMAX, and ROWMAX is its absolute value.
				//                 Determine both ROWMAX and JMAX.
				if imax != k {
					jmax = imax + goblas.Idamax(k-imax, w.Vector(imax+1-1, kw-1-1), 1)
					rowmax = math.Abs(w.Get(jmax-1, kw-1-1))
				} else {
					rowmax = zero
				}

				if imax > 1 {
					itemp = goblas.Idamax(imax-1, w.Vector(0, kw-1-1), 1)
					dtemp = math.Abs(w.Get(itemp-1, kw-1-1))
					if dtemp > rowmax {
						rowmax = dtemp
						jmax = itemp
					}
				}

				//                 Equivalent to testing for
				//                 ABS( W( IMAX, KW-1 ) ).GE.ALPHA*ROWMAX
				//                 (used to handle NaN and Inf)
				if !(math.Abs(w.Get(imax-1, kw-1-1)) < alpha*rowmax) {
					//                    interchange rows and columns K and IMAX,
					//                    use 1-by-1 pivot block
					kp = imax

					//                    copy column KW-1 of W to column KW of W
					goblas.Dcopy(k, w.Vector(0, kw-1-1), 1, w.Vector(0, kw-1), 1)

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
					goblas.Dcopy(k, w.Vector(0, kw-1-1), 1, w.Vector(0, kw-1), 1)

				}

				//                 End pivot search loop body
				if !done {
					goto label12
				}

			}

			//           ============================================================
			kk = k - kstep + 1

			//           KKW is the column of W which corresponds to column KK of A
			kkw = (*nb) + kk - (*n)

			if (kstep == 2) && (p != k) {
				//              Copy non-updated column K to column P
				goblas.Dcopy(k-p, a.Vector(p+1-1, k-1), 1, a.Vector(p-1, p+1-1), *lda)
				goblas.Dcopy(p, a.Vector(0, k-1), 1, a.Vector(0, p-1), 1)

				//              Interchange rows K and P in last N-K+1 columns of A
				//              and last N-K+2 columns of W
				goblas.Dswap((*n)-k+1, a.Vector(k-1, k-1), *lda, a.Vector(p-1, k-1), *lda)
				goblas.Dswap((*n)-kk+1, w.Vector(k-1, kkw-1), *ldw, w.Vector(p-1, kkw-1), *ldw)
			}

			//           Updated column KP is already stored in column KKW of W
			if kp != kk {
				//              Copy non-updated column KK to column KP
				a.Set(kp-1, k-1, a.Get(kk-1, k-1))
				goblas.Dcopy(k-1-kp, a.Vector(kp+1-1, kk-1), 1, a.Vector(kp-1, kp+1-1), *lda)
				goblas.Dcopy(kp, a.Vector(0, kk-1), 1, a.Vector(0, kp-1), 1)

				//              Interchange rows KK and KP in last N-KK+1 columns
				//              of A and W
				goblas.Dswap((*n)-kk+1, a.Vector(kk-1, kk-1), *lda, a.Vector(kp-1, kk-1), *lda)
				goblas.Dswap((*n)-kk+1, w.Vector(kk-1, kkw-1), *ldw, w.Vector(kp-1, kkw-1), *ldw)
			}

			if kstep == 1 {
				//              1-by-1 pivot block D(k): column KW of W now holds
				//
				//              W(k) = U(k)*D(k)
				//
				//              where U(k) is the k-th column of U
				//
				//              Store U(k) in column k of A
				goblas.Dcopy(k, w.Vector(0, kw-1), 1, a.Vector(0, k-1), 1)
				if k > 1 {
					if math.Abs(a.Get(k-1, k-1)) >= sfmin {
						r1 = one / a.Get(k-1, k-1)
						goblas.Dscal(k-1, r1, a.Vector(0, k-1), 1)
					} else if a.Get(k-1, k-1) != zero {
						for ii = 1; ii <= k-1; ii++ {
							a.Set(ii-1, k-1, a.Get(ii-1, k-1)/a.Get(k-1, k-1))
						}
					}
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
					t = one / (d11*d22 - one)
					for j = 1; j <= k-2; j++ {
						a.Set(j-1, k-1-1, t*((d11*w.Get(j-1, kw-1-1)-w.Get(j-1, kw-1))/d12))
						a.Set(j-1, k-1, t*((d22*w.Get(j-1, kw-1)-w.Get(j-1, kw-1-1))/d12))
					}
				}

				//              Copy D(k) to A
				a.Set(k-1-1, k-1-1, w.Get(k-1-1, kw-1-1))
				a.Set(k-1-1, k-1, w.Get(k-1-1, kw-1))
				a.Set(k-1, k-1, w.Get(k-1, kw-1))
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

	label30:
		;

		//        Update the upper triangle of A11 (= A(1:k,1:k)) as
		//
		//        A11 := A11 - U12*D*U12**T = A11 - U12*W**T
		//
		//        computing blocks of NB columns at a time
		for j = ((k-1)/(*nb))*(*nb) + 1; j >= 1; j -= (*nb) {
			jb = minint(*nb, k-j+1)

			//           Update the upper triangle of the diagonal block
			for jj = j; jj <= j+jb-1; jj++ {
				err = goblas.Dgemv(NoTrans, jj-j+1, (*n)-k, -one, a.Off(j-1, k+1-1), *lda, w.Vector(jj-1, kw+1-1), *ldw, one, a.Vector(j-1, jj-1), 1)
			}

			//           Update the rectangular superdiagonal block
			if j >= 2 {
				err = goblas.Dgemm(mat.NoTrans, mat.Trans, j-1, jb, (*n)-k, -one, a.Off(0, k+1-1), *lda, w.Off(j-1, kw+1-1), *ldw, one, a.Off(0, j-1), *lda)
			}
		}

		//        Put U12 in standard form by partially undoing the interchanges
		//        in columns k+1:n
		j = k + 1
	label60:
		;

		kstep = 1
		jp1 = 1
		jj = j
		jp2 = (*ipiv)[j-1]
		if jp2 < 0 {
			jp2 = -jp2
			j = j + 1
			jp1 = -(*ipiv)[j-1]
			kstep = 2
		}

		j = j + 1
		if jp2 != jj && j <= (*n) {
			goblas.Dswap((*n)-j+1, a.Vector(jp2-1, j-1), *lda, a.Vector(jj-1, j-1), *lda)
		}
		jj = j - 1
		if jp1 != jj && kstep == 2 {
			goblas.Dswap((*n)-j+1, a.Vector(jp1-1, j-1), *lda, a.Vector(jj-1, j-1), *lda)
		}
		if j <= (*n) {
			goto label60
		}

		//        Set KB to the number of columns factorized
		(*kb) = (*n) - k

	} else {
		//        Factorize the leading columns of A using the lower triangle
		//        of A and working forwards, and compute the matrix W = L21*D
		//        for use in updating A22
		//
		//        K is the main loop index, increasing from 1 in steps of 1 or 2
		k = 1
	label70:
		;

		//        Exit from loop
		if (k >= (*nb) && (*nb) < (*n)) || k > (*n) {
			goto label90
		}

		kstep = 1
		p = k

		//        Copy column K of A to column K of W and update it
		goblas.Dcopy((*n)-k+1, a.Vector(k-1, k-1), 1, w.Vector(k-1, k-1), 1)
		if k > 1 {
			err = goblas.Dgemv(NoTrans, (*n)-k+1, k-1, -one, a.Off(k-1, 0), *lda, w.Vector(k-1, 0), *ldw, one, w.Vector(k-1, k-1), 1)
		}

		//        Determine rows and columns to be interchanged and whether
		//        a 1-by-1 or 2-by-2 pivot block will be used
		absakk = math.Abs(w.Get(k-1, k-1))

		//        IMAX is the row-index of the largest off-diagonal element in
		//        column K, and COLMAX is its absolute value.
		//        Determine both COLMAX and IMAX.
		if k < (*n) {
			imax = k + goblas.Idamax((*n)-k, w.Vector(k+1-1, k-1), 1)
			colmax = math.Abs(w.Get(imax-1, k-1))
		} else {
			colmax = zero
		}

		if maxf64(absakk, colmax) == zero {
			//           Column K is zero or underflow: set INFO and continue
			if (*info) == 0 {
				(*info) = k
			}
			kp = k
			goblas.Dcopy((*n)-k+1, w.Vector(k-1, k-1), 1, a.Vector(k-1, k-1), 1)
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
				goblas.Dcopy(imax-k, a.Vector(imax-1, k-1), *lda, w.Vector(k-1, k+1-1), 1)
				goblas.Dcopy((*n)-imax+1, a.Vector(imax-1, imax-1), 1, w.Vector(imax-1, k+1-1), 1)
				if k > 1 {
					err = goblas.Dgemv(NoTrans, (*n)-k+1, k-1, -one, a.Off(k-1, 0), *lda, w.Vector(imax-1, 0), *ldw, one, w.Vector(k-1, k+1-1), 1)
				}

				//                 JMAX is the column-index of the largest off-diagonal
				//                 element in row IMAX, and ROWMAX is its absolute value.
				//                 Determine both ROWMAX and JMAX.
				if imax != k {
					jmax = k - 1 + goblas.Idamax(imax-k, w.Vector(k-1, k+1-1), 1)
					rowmax = math.Abs(w.Get(jmax-1, k+1-1))
				} else {
					rowmax = zero
				}

				if imax < (*n) {
					itemp = imax + goblas.Idamax((*n)-imax, w.Vector(imax+1-1, k+1-1), 1)
					dtemp = math.Abs(w.Get(itemp-1, k+1-1))
					if dtemp > rowmax {
						rowmax = dtemp
						jmax = itemp
					}
				}

				//                 Equivalent to testing for
				//                 ABS( W( IMAX, K+1 ) ).GE.ALPHA*ROWMAX
				//                 (used to handle NaN and Inf)
				if !(math.Abs(w.Get(imax-1, k+1-1)) < alpha*rowmax) {
					//                    interchange rows and columns K and IMAX,
					//                    use 1-by-1 pivot block
					kp = imax

					//                    copy column K+1 of W to column K of W
					goblas.Dcopy((*n)-k+1, w.Vector(k-1, k+1-1), 1, w.Vector(k-1, k-1), 1)

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
					goblas.Dcopy((*n)-k+1, w.Vector(k-1, k+1-1), 1, w.Vector(k-1, k-1), 1)

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
				goblas.Dcopy(p-k, a.Vector(k-1, k-1), 1, a.Vector(p-1, k-1), *lda)
				goblas.Dcopy((*n)-p+1, a.Vector(p-1, k-1), 1, a.Vector(p-1, p-1), 1)

				//              Interchange rows K and P in first K columns of A
				//              and first K+1 columns of W
				goblas.Dswap(k, a.Vector(k-1, 0), *lda, a.Vector(p-1, 0), *lda)
				goblas.Dswap(kk, w.Vector(k-1, 0), *ldw, w.Vector(p-1, 0), *ldw)
			}

			//           Updated column KP is already stored in column KK of W
			if kp != kk {
				//              Copy non-updated column KK to column KP
				a.Set(kp-1, k-1, a.Get(kk-1, k-1))
				goblas.Dcopy(kp-k-1, a.Vector(k+1-1, kk-1), 1, a.Vector(kp-1, k+1-1), *lda)
				goblas.Dcopy((*n)-kp+1, a.Vector(kp-1, kk-1), 1, a.Vector(kp-1, kp-1), 1)

				//              Interchange rows KK and KP in first KK columns of A and W
				goblas.Dswap(kk, a.Vector(kk-1, 0), *lda, a.Vector(kp-1, 0), *lda)
				goblas.Dswap(kk, w.Vector(kk-1, 0), *ldw, w.Vector(kp-1, 0), *ldw)
			}

			if kstep == 1 {
				//              1-by-1 pivot block D(k): column k of W now holds
				//
				//              W(k) = L(k)*D(k)
				//
				//              where L(k) is the k-th column of L
				//
				//              Store L(k) in column k of A
				goblas.Dcopy((*n)-k+1, w.Vector(k-1, k-1), 1, a.Vector(k-1, k-1), 1)
				if k < (*n) {
					if math.Abs(a.Get(k-1, k-1)) >= sfmin {
						r1 = one / a.Get(k-1, k-1)
						goblas.Dscal((*n)-k, r1, a.Vector(k+1-1, k-1), 1)
					} else if a.Get(k-1, k-1) != zero {
						for ii = k + 1; ii <= (*n); ii++ {
							a.Set(ii-1, k-1, a.Get(ii-1, k-1)/a.Get(k-1, k-1))
						}
					}
				}

			} else {
				//              2-by-2 pivot block D(k): columns k and k+1 of W now hold
				//
				//              ( W(k) W(k+1) ) = ( L(k) L(k+1) )*D(k)
				//
				//              where L(k) and L(k+1) are the k-th and (k+1)-th columns
				//              of L
				if k < (*n)-1 {
					//                 Store L(k) and L(k+1) in columns k and k+1 of A
					d21 = w.Get(k+1-1, k-1)
					d11 = w.Get(k+1-1, k+1-1) / d21
					d22 = w.Get(k-1, k-1) / d21
					t = one / (d11*d22 - one)
					for j = k + 2; j <= (*n); j++ {
						a.Set(j-1, k-1, t*((d11*w.Get(j-1, k-1)-w.Get(j-1, k+1-1))/d21))
						a.Set(j-1, k+1-1, t*((d22*w.Get(j-1, k+1-1)-w.Get(j-1, k-1))/d21))
					}
				}

				//              Copy D(k) to A
				a.Set(k-1, k-1, w.Get(k-1, k-1))
				a.Set(k+1-1, k-1, w.Get(k+1-1, k-1))
				a.Set(k+1-1, k+1-1, w.Get(k+1-1, k+1-1))
			}
		}

		//        Store details of the interchanges in IPIV
		if kstep == 1 {
			(*ipiv)[k-1] = kp
		} else {
			(*ipiv)[k-1] = -p
			(*ipiv)[k+1-1] = -kp
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
		for j = k; j <= (*n); j += (*nb) {
			jb = minint(*nb, (*n)-j+1)

			//           Update the lower triangle of the diagonal block
			for jj = j; jj <= j+jb-1; jj++ {
				err = goblas.Dgemv(NoTrans, j+jb-jj, k-1, -one, a.Off(jj-1, 0), *lda, w.Vector(jj-1, 0), *ldw, one, a.Vector(jj-1, jj-1), 1)
			}

			//           Update the rectangular subdiagonal block
			if j+jb <= (*n) {
				err = goblas.Dgemm(mat.NoTrans, mat.Trans, (*n)-j-jb+1, jb, k-1, -one, a.Off(j+jb-1, 0), *lda, w.Off(j-1, 0), *ldw, one, a.Off(j+jb-1, j-1), *lda)
			}
		}

		//        Put L21 in standard form by partially undoing the interchanges
		//        in columns 1:k-1
		j = k - 1
	label120:
		;

		kstep = 1
		jp1 = 1
		jj = j
		jp2 = (*ipiv)[j-1]
		if jp2 < 0 {
			jp2 = -jp2
			j = j - 1
			jp1 = -(*ipiv)[j-1]
			kstep = 2
		}

		j = j - 1
		if jp2 != jj && j >= 1 {
			goblas.Dswap(j, a.Vector(jp2-1, 0), *lda, a.Vector(jj-1, 0), *lda)
		}
		jj = j + 1
		if jp1 != jj && kstep == 2 {
			goblas.Dswap(j, a.Vector(jp1-1, 0), *lda, a.Vector(jj-1, 0), *lda)
		}
		if j >= 1 {
			goto label120
		}

		//        Set KB to the number of columns factorized
		(*kb) = k - 1

	}
}
