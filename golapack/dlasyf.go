package golapack

import (
	"golinalg/goblas"
	"golinalg/mat"
	"math"
)

// Dlasyf computes a partial factorization of a real symmetric matrix A
// using the Bunch-Kaufman diagonal pivoting method. The partial
// factorization has the form:
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
// DLASYF is an auxiliary routine called by DSYTRF. It uses blocked code
// (calling Level 3 BLAS) to update the submatrix A11 (if UPLO = 'U') or
// A22 (if UPLO = 'L').
func Dlasyf(uplo byte, n, nb, kb *int, a *mat.Matrix, lda *int, ipiv *[]int, w *mat.Matrix, ldw, info *int) {
	var absakk, alpha, colmax, d11, d21, d22, eight, one, r1, rowmax, sevten, t, zero float64
	var imax, j, jb, jj, jmax, jp, k, kk, kkw, kp, kstep, kw int

	zero = 0.0
	one = 1.0
	eight = 8.0
	sevten = 17.0

	(*info) = 0

	//     Initialize ALPHA for use in choosing pivot block size.
	alpha = (one + math.Sqrt(sevten)) / eight

	if uplo == 'U' {
		//        Factorize the trailing columns of A using the upper triangle
		//        of A and working backwards, and compute the matrix W = U12*D
		//        for use in updating A11
		//
		//        K is the main loop index, decreasing from N in steps of 1 or 2
		//
		//        KW is the column of W which corresponds to column K of A
		k = (*n)
	label10:
		;
		kw = (*nb) + k - (*n)

		//        Exit from loop
		if (k <= (*n)-(*nb)+1 && (*nb) < (*n)) || k < 1 {
			goto label30
		}

		//        Copy column K of A to column KW of W and update it
		goblas.Dcopy(&k, a.Vector(0, k-1), toPtr(1), w.Vector(0, kw-1), toPtr(1))
		if k < (*n) {
			goblas.Dgemv(mat.NoTrans, &k, toPtr((*n)-k), toPtrf64(-one), a.Off(0, k+1-1), lda, w.Vector(k-1, kw+1-1), ldw, &one, w.Vector(0, kw-1), toPtr(1))
		}

		kstep = 1

		//        Determine rows and columns to be interchanged and whether
		//        a 1-by-1 or 2-by-2 pivot block will be used
		absakk = math.Abs(w.Get(k-1, kw-1))

		//        IMAX is the row-index of the largest off-diagonal element in
		//        column K, and COLMAX is its absolute value.
		//        Determine both COLMAX and IMAX.
		if k > 1 {
			imax = goblas.Idamax(toPtr(k-1), w.Vector(0, kw-1), toPtr(1))
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
		} else {
			if absakk >= alpha*colmax {
				//              no interchange, use 1-by-1 pivot block
				kp = k
			} else {
				//              Copy column IMAX to column KW-1 of W and update it
				goblas.Dcopy(&imax, a.Vector(0, imax-1), toPtr(1), w.Vector(0, kw-1-1), toPtr(1))
				goblas.Dcopy(toPtr(k-imax), a.Vector(imax-1, imax+1-1), lda, w.Vector(imax+1-1, kw-1-1), toPtr(1))
				if k < (*n) {
					goblas.Dgemv(mat.NoTrans, &k, toPtr((*n)-k), toPtrf64(-one), a.Off(0, k+1-1), lda, w.Vector(imax-1, kw+1-1), ldw, &one, w.Vector(0, kw-1-1), toPtr(1))
				}

				//              JMAX is the column-index of the largest off-diagonal
				//              element in row IMAX, and ROWMAX is its absolute value
				jmax = imax + goblas.Idamax(toPtr(k-imax), w.Vector(imax+1-1, kw-1-1), toPtr(1))
				rowmax = math.Abs(w.Get(jmax-1, kw-1-1))
				if imax > 1 {
					jmax = goblas.Idamax(toPtr(imax-1), w.Vector(0, kw-1-1), toPtr(1))
					rowmax = maxf64(rowmax, math.Abs(w.Get(jmax-1, kw-1-1)))
				}

				if absakk >= alpha*colmax*(colmax/rowmax) {
					//                 no interchange, use 1-by-1 pivot block
					kp = k
				} else if math.Abs(w.Get(imax-1, kw-1-1)) >= alpha*rowmax {
					//                 interchange rows and columns K and IMAX, use 1-by-1
					//                 pivot block
					kp = imax

					//                 copy column KW-1 of W to column KW of W
					goblas.Dcopy(&k, w.Vector(0, kw-1-1), toPtr(1), w.Vector(0, kw-1), toPtr(1))
				} else {
					//                 interchange rows and columns K-1 and IMAX, use 2-by-2
					//                 pivot block
					kp = imax
					kstep = 2
				}
			}

			//           ============================================================
			//
			//           KK is the column of A where pivoting step stopped
			kk = k - kstep + 1

			//           KKW is the column of W which corresponds to column KK of A
			kkw = (*nb) + kk - (*n)

			//           Interchange rows and columns KP and KK.
			//           Updated column KP is already stored in column KKW of W.
			if kp != kk {
				//              Copy non-updated column KK to column KP of submatrix A
				//              at step K. No need to copy element into column K
				//              (or K and K-1 for 2-by-2 pivot) of A, since these columns
				//              will be later overwritten.
				a.Set(kp-1, kp-1, a.Get(kk-1, kk-1))
				goblas.Dcopy(toPtr(kk-1-kp), a.Vector(kp+1-1, kk-1), toPtr(1), a.Vector(kp-1, kp+1-1), lda)
				if kp > 1 {
					goblas.Dcopy(toPtr(kp-1), a.Vector(0, kk-1), toPtr(1), a.Vector(0, kp-1), toPtr(1))
				}

				//              Interchange rows KK and KP in last K+1 to N columns of A
				//              (columns K (or K and K-1 for 2-by-2 pivot) of A will be
				//              later overwritten). Interchange rows KK and KP
				//              in last KKW to NB columns of W.
				if k < (*n) {
					goblas.Dswap(toPtr((*n)-k), a.Vector(kk-1, k+1-1), lda, a.Vector(kp-1, k+1-1), lda)
				}
				goblas.Dswap(toPtr((*n)-kk+1), w.Vector(kk-1, kkw-1), ldw, w.Vector(kp-1, kkw-1), ldw)
			}

			if kstep == 1 {
				//              1-by-1 pivot block D(k): column kw of W now holds
				//
				//              W(kw) = U(k)*D(k),
				//
				//              where U(k) is the k-th column of U
				//
				//              Store subdiag. elements of column U(k)
				//              and 1-by-1 block D(k) in column k of A.
				//              NOTE: Diagonal element U(k,k) is a UNIT element
				//              and not stored.
				//                 A(k,k) := D(k,k) = W(k,kw)
				//                 A(1:k-1,k) := U(1:k-1,k) = W(1:k-1,kw)/D(k,k)
				goblas.Dcopy(&k, w.Vector(0, kw-1), toPtr(1), a.Vector(0, k-1), toPtr(1))
				r1 = one / a.Get(k-1, k-1)
				goblas.Dscal(toPtr(k-1), &r1, a.Vector(0, k-1), toPtr(1))

			} else {
				//              2-by-2 pivot block D(k): columns kw and kw-1 of W now hold
				//
				//              ( W(kw-1) W(kw) ) = ( U(k-1) U(k) )*D(k)
				//
				//              where U(k) and U(k-1) are the k-th and (k-1)-th columns
				//              of U
				//
				//              Store U(1:k-2,k-1) and U(1:k-2,k) and 2-by-2
				//              block D(k-1:k,k-1:k) in columns k-1 and k of A.
				//              NOTE: 2-by-2 diagonal block U(k-1:k,k-1:k) is a UNIT
				//              block and not stored.
				//                 A(k-1:k,k-1:k) := D(k-1:k,k-1:k) = W(k-1:k,kw-1:kw)
				//                 A(1:k-2,k-1:k) := U(1:k-2,k:k-1:k) =
				//                 = W(1:k-2,kw-1:kw) * ( D(k-1:k,k-1:k)**(-1) )

				if k > 2 {
					//                 Compose the columns of the inverse of 2-by-2 pivot
					//                 block D in the following way to reduce the number
					//                 of FLOPS when we myltiply panel ( W(kw-1) W(kw) ) by
					//                 this inverse
					//
					//                 D**(-1) = ( d11 d21 )**(-1) =
					//                           ( d21 d22 )
					//
					//                 = 1/(d11*d22-d21**2) * ( ( d22 ) (-d21 ) ) =
					//                                        ( (-d21 ) ( d11 ) )
					//
					//                 = 1/d21 * 1/((d11/d21)*(d22/d21)-1) *
					//
					//                   * ( ( d22/d21 ) (      -1 ) ) =
					//                     ( (      -1 ) ( d11/d21 ) )
					//
					//                 = 1/d21 * 1/(D22*D11-1) * ( ( D11 ) (  -1 ) ) =
					//                                           ( ( -1  ) ( D22 ) )
					//
					//                 = 1/d21 * T * ( ( D11 ) (  -1 ) )
					//                               ( (  -1 ) ( D22 ) )
					//
					//                 = D21 * ( ( D11 ) (  -1 ) )
					//                         ( (  -1 ) ( D22 ) )
					d21 = w.Get(k-1-1, kw-1)
					d11 = w.Get(k-1, kw-1) / d21
					d22 = w.Get(k-1-1, kw-1-1) / d21
					t = one / (d11*d22 - one)
					d21 = t / d21

					//                 Update elements in columns A(k-1) and A(k) as
					//                 dot products of rows of ( W(kw-1) W(kw) ) and columns
					//                 of D**(-1)
					for j = 1; j <= k-2; j++ {
						a.Set(j-1, k-1-1, d21*(d11*w.Get(j-1, kw-1-1)-w.Get(j-1, kw-1)))
						a.Set(j-1, k-1, d21*(d22*w.Get(j-1, kw-1)-w.Get(j-1, kw-1-1)))
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
		//        A11 := A11 - U12*D*U12**T = A11 - U12*W**T
		//
		//        computing blocks of NB columns at a time
		for j = ((k-1)/(*nb))*(*nb) + 1; j >= 1; j -= (*nb) {
			jb = minint(*nb, k-j+1)

			//           Update the upper triangle of the diagonal block
			for jj = j; jj <= j+jb-1; jj++ {
				goblas.Dgemv(mat.NoTrans, toPtr(jj-j+1), toPtr((*n)-k), toPtrf64(-one), a.Off(j-1, k+1-1), lda, w.Vector(jj-1, kw+1-1), ldw, &one, a.Vector(j-1, jj-1), toPtr(1))
			}

			//           Update the rectangular superdiagonal block
			goblas.Dgemm(mat.NoTrans, mat.Trans, toPtr(j-1), &jb, toPtr((*n)-k), toPtrf64(-one), a.Off(0, k+1-1), lda, w.Off(j-1, kw+1-1), ldw, &one, a.Off(0, j-1), lda)
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
		if jp != jj && j <= (*n) {
			goblas.Dswap(toPtr((*n)-j+1), a.Vector(jp-1, j-1), lda, a.Vector(jj-1, j-1), lda)
		}
		if j < (*n) {
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

		//        Copy column K of A to column K of W and update it
		goblas.Dcopy(toPtr((*n)-k+1), a.Vector(k-1, k-1), toPtr(1), w.Vector(k-1, k-1), toPtr(1))
		goblas.Dgemv(mat.NoTrans, toPtr((*n)-k+1), toPtr(k-1), toPtrf64(-one), a.Off(k-1, 0), lda, w.Vector(k-1, 0), ldw, &one, w.Vector(k-1, k-1), toPtr(1))

		kstep = 1

		//        Determine rows and columns to be interchanged and whether
		//        a 1-by-1 or 2-by-2 pivot block will be used
		absakk = math.Abs(w.Get(k-1, k-1))

		//        IMAX is the row-index of the largest off-diagonal element in
		//        column K, and COLMAX is its absolute value.
		//        Determine both COLMAX and IMAX.
		if k < (*n) {
			imax = k + goblas.Idamax(toPtr((*n)-k), w.Vector(k+1-1, k-1), toPtr(1))
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
		} else {
			if absakk >= alpha*colmax {
				//              no interchange, use 1-by-1 pivot block
				kp = k
			} else {
				//              Copy column IMAX to column K+1 of W and update it
				goblas.Dcopy(toPtr(imax-k), a.Vector(imax-1, k-1), lda, w.Vector(k-1, k+1-1), toPtr(1))
				goblas.Dcopy(toPtr((*n)-imax+1), a.Vector(imax-1, imax-1), toPtr(1), w.Vector(imax-1, k+1-1), toPtr(1))
				goblas.Dgemv(mat.NoTrans, toPtr((*n)-k+1), toPtr(k-1), toPtrf64(-one), a.Off(k-1, 0), lda, w.Vector(imax-1, 0), ldw, &one, w.Vector(k-1, k+1-1), toPtr(1))

				//              JMAX is the column-index of the largest off-diagonal
				//              element in row IMAX, and ROWMAX is its absolute value
				jmax = k - 1 + goblas.Idamax(toPtr(imax-k), w.Vector(k-1, k+1-1), toPtr(1))
				rowmax = math.Abs(w.Get(jmax-1, k+1-1))
				if imax < (*n) {
					jmax = imax + goblas.Idamax(toPtr((*n)-imax), w.Vector(imax+1-1, k+1-1), toPtr(1))
					rowmax = maxf64(rowmax, math.Abs(w.Get(jmax-1, k+1-1)))
				}

				if absakk >= alpha*colmax*(colmax/rowmax) {
					//                 no interchange, use 1-by-1 pivot block
					kp = k
				} else if math.Abs(w.Get(imax-1, k+1-1)) >= alpha*rowmax {
					//                 interchange rows and columns K and IMAX, use 1-by-1
					//                 pivot block
					kp = imax

					//                 copy column K+1 of W to column K of W
					goblas.Dcopy(toPtr((*n)-k+1), w.Vector(k-1, k+1-1), toPtr(1), w.Vector(k-1, k-1), toPtr(1))
				} else {
					//                 interchange rows and columns K+1 and IMAX, use 2-by-2
					//                 pivot block
					kp = imax
					kstep = 2
				}
			}

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
				a.Set(kp-1, kp-1, a.Get(kk-1, kk-1))
				goblas.Dcopy(toPtr(kp-kk-1), a.Vector(kk+1-1, kk-1), toPtr(1), a.Vector(kp-1, kk+1-1), lda)
				if kp < (*n) {
					goblas.Dcopy(toPtr((*n)-kp), a.Vector(kp+1-1, kk-1), toPtr(1), a.Vector(kp+1-1, kp-1), toPtr(1))
				}

				//              Interchange rows KK and KP in first K-1 columns of A
				//              (columns K (or K and K+1 for 2-by-2 pivot) of A will be
				//              later overwritten). Interchange rows KK and KP
				//              in first KK columns of W.
				if k > 1 {
					goblas.Dswap(toPtr(k-1), a.Vector(kk-1, 0), lda, a.Vector(kp-1, 0), lda)
				}
				goblas.Dswap(&kk, w.Vector(kk-1, 0), ldw, w.Vector(kp-1, 0), ldw)
			}

			if kstep == 1 {
				//              1-by-1 pivot block D(k): column k of W now holds
				//
				//              W(k) = L(k)*D(k),
				//
				//              where L(k) is the k-th column of L
				//
				//              Store subdiag. elements of column L(k)
				//              and 1-by-1 block D(k) in column k of A.
				//              (NOTE: Diagonal element L(k,k) is a UNIT element
				//              and not stored)
				//                 A(k,k) := D(k,k) = W(k,k)
				//                 A(k+1:N,k) := L(k+1:N,k) = W(k+1:N,k)/D(k,k)
				goblas.Dcopy(toPtr((*n)-k+1), w.Vector(k-1, k-1), toPtr(1), a.Vector(k-1, k-1), toPtr(1))
				if k < (*n) {
					r1 = one / a.Get(k-1, k-1)
					goblas.Dscal(toPtr((*n)-k), &r1, a.Vector(k+1-1, k-1), toPtr(1))
				}

			} else {
				//              2-by-2 pivot block D(k): columns k and k+1 of W now hold
				//
				//              ( W(k) W(k+1) ) = ( L(k) L(k+1) )*D(k)
				//
				//              where L(k) and L(k+1) are the k-th and (k+1)-th columns
				//              of L
				//
				//              Store L(k+2:N,k) and L(k+2:N,k+1) and 2-by-2
				//              block D(k:k+1,k:k+1) in columns k and k+1 of A.
				//              (NOTE: 2-by-2 diagonal block L(k:k+1,k:k+1) is a UNIT
				//              block and not stored)
				//                 A(k:k+1,k:k+1) := D(k:k+1,k:k+1) = W(k:k+1,k:k+1)
				//                 A(k+2:N,k:k+1) := L(k+2:N,k:k+1) =
				//                 = W(k+2:N,k:k+1) * ( D(k:k+1,k:k+1)**(-1) )
				if k < (*n)-1 {
					//                 Compose the columns of the inverse of 2-by-2 pivot
					//                 block D in the following way to reduce the number
					//                 of FLOPS when we myltiply panel ( W(k) W(k+1) ) by
					//                 this inverse
					//
					//                 D**(-1) = ( d11 d21 )**(-1) =
					//                           ( d21 d22 )
					//
					//                 = 1/(d11*d22-d21**2) * ( ( d22 ) (-d21 ) ) =
					//                                        ( (-d21 ) ( d11 ) )
					//
					//                 = 1/d21 * 1/((d11/d21)*(d22/d21)-1) *
					//
					//                   * ( ( d22/d21 ) (      -1 ) ) =
					//                     ( (      -1 ) ( d11/d21 ) )
					//
					//                 = 1/d21 * 1/(D22*D11-1) * ( ( D11 ) (  -1 ) ) =
					//                                           ( ( -1  ) ( D22 ) )
					//
					//                 = 1/d21 * T * ( ( D11 ) (  -1 ) )
					//                               ( (  -1 ) ( D22 ) )
					//
					//                 = D21 * ( ( D11 ) (  -1 ) )
					//                         ( (  -1 ) ( D22 ) )
					d21 = w.Get(k+1-1, k-1)
					d11 = w.Get(k+1-1, k+1-1) / d21
					d22 = w.Get(k-1, k-1) / d21
					t = one / (d11*d22 - one)
					d21 = t / d21

					//                 Update elements in columns A(k) and A(k+1) as
					//                 dot products of rows of ( W(k) W(k+1) ) and columns
					//                 of D**(-1)
					for j = k + 2; j <= (*n); j++ {
						a.Set(j-1, k-1, d21*(d11*w.Get(j-1, k-1)-w.Get(j-1, k+1-1)))
						a.Set(j-1, k+1-1, d21*(d22*w.Get(j-1, k+1-1)-w.Get(j-1, k-1)))
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
			(*ipiv)[k-1] = -kp
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
				goblas.Dgemv(mat.NoTrans, toPtr(j+jb-jj), toPtr(k-1), toPtrf64(-one), a.Off(jj-1, 0), lda, w.Vector(jj-1, 0), ldw, &one, a.Vector(jj-1, jj-1), toPtr(1))
			}

			//           Update the rectangular subdiagonal block
			if j+jb <= (*n) {
				goblas.Dgemm(mat.NoTrans, mat.Trans, toPtr((*n)-j-jb+1), &jb, toPtr(k-1), toPtrf64(-one), a.Off(j+jb-1, 0), lda, w.Off(j-1, 0), ldw, &one, a.Off(j+jb-1, j-1), lda)
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
			goblas.Dswap(&j, a.Vector(jp-1, 0), lda, a.Vector(jj-1, 0), lda)
		}
		if j > 1 {
			goto label120
		}

		//        Set KB to the number of columns factorized
		(*kb) = k - 1

	}
}
