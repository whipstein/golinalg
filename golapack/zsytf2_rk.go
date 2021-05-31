package golapack

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zsytf2rk computes the factorization of a complex symmetric matrix A
// using the bounded Bunch-Kaufman (rook) diagonal pivoting method:
//
//    A = P*U*D*(U**T)*(P**T) or A = P*L*D*(L**T)*(P**T),
//
// where U (or L) is unit upper (or lower) triangular matrix,
// U**T (or L**T) is the transpose of U (or L), P is a permutation
// matrix, P**T is the transpose of P, and D is symmetric and block
// diagonal with 1-by-1 and 2-by-2 diagonal blocks.
//
// This is the unblocked version of the algorithm, calling Level 2 BLAS.
// For more information see Further Details section.
func Zsytf2rk(uplo byte, n *int, a *mat.CMatrix, lda *int, e *mat.CVector, ipiv *[]int, info *int) {
	var done, upper bool
	var cone, czero, d11, d12, d21, d22, t, wk, wkm1, wkp1 complex128
	var absakk, alpha, colmax, dtemp, eight, one, rowmax, sevten, sfmin, zero float64
	var i, ii, imax, itemp, j, jmax, k, kk, kp, kstep, p int

	zero = 0.0
	one = 1.0
	eight = 8.0
	sevten = 17.0
	cone = (1.0 + 0.0*1i)
	czero = (0.0 + 0.0*1i)

	Cabs1 := func(z complex128) float64 { return math.Abs(real(z)) + math.Abs(imag(z)) }

	//     Test the input parameters.
	(*info) = 0
	upper = uplo == 'U'
	if !upper && uplo != 'L' {
		(*info) = -1
	} else if (*n) < 0 {
		(*info) = -2
	} else if (*lda) < maxint(1, *n) {
		(*info) = -4
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("ZSYTF2_RK"), -(*info))
		return
	}

	//     Initialize ALPHA for use in choosing pivot block size.
	alpha = (one + math.Sqrt(sevten)) / eight

	//     Compute machine safe minimum
	sfmin = Dlamch(SafeMinimum)

	if upper {
		//        Factorize A as U*D*U**T using the upper triangle of A
		//
		//        Initialize the first entry of array E, where superdiagonal
		//        elements of D are stored
		e.Set(0, czero)

		//        K is the main loop index, decreasing from N to 1 in steps of
		//        1 or 2
		k = (*n)
	label10:
		;

		//        If K < 1, exit from loop
		if k < 1 {
			goto label34
		}
		kstep = 1
		p = k

		//        Determine rows and columns to be interchanged and whether
		//        a 1-by-1 or 2-by-2 pivot block will be used
		absakk = Cabs1(a.Get(k-1, k-1))

		//        IMAX is the row-index of the largest off-diagonal element in
		//        column K, and COLMAX is its absolute value.
		//        Determine both COLMAX and IMAX.
		if k > 1 {
			imax = goblas.Izamax(toPtr(k-1), a.CVector(0, k-1), func() *int { y := 1; return &y }())
			colmax = Cabs1(a.Get(imax-1, k-1))
		} else {
			colmax = zero
		}

		if maxf64(absakk, colmax) == zero {
			//           Column K is zero or underflow: set INFO and continue
			if (*info) == 0 {
				(*info) = k
			}
			kp = k

			//           Set E( K ) to zero
			if k > 1 {
				e.Set(k-1, czero)
			}

		} else {
			//           Test for interchange
			//
			//           Equivalent to testing for (used to handle NaN and Inf)
			//           ABSAKK.GE.ALPHA*COLMAX
			if !(absakk < alpha*colmax) {
				//              no interchange,
				//              use 1-by-1 pivot block
				kp = k
			} else {

				done = false

				//              Loop until pivot found
			label12:
				;

				//                 Begin pivot search loop body
				//
				//                 JMAX is the column-index of the largest off-diagonal
				//                 element in row IMAX, and ROWMAX is its absolute value.
				//                 Determine both ROWMAX and JMAX.
				if imax != k {
					jmax = imax + goblas.Izamax(toPtr(k-imax), a.CVector(imax-1, imax+1-1), lda)
					rowmax = Cabs1(a.Get(imax-1, jmax-1))
				} else {
					rowmax = zero
				}

				if imax > 1 {
					itemp = goblas.Izamax(toPtr(imax-1), a.CVector(0, imax-1), func() *int { y := 1; return &y }())
					dtemp = Cabs1(a.Get(itemp-1, imax-1))
					if dtemp > rowmax {
						rowmax = dtemp
						jmax = itemp
					}
				}

				//                 Equivalent to testing for (used to handle NaN and Inf)
				//                 ABS( A( IMAX, IMAX ) ).GE.ALPHA*ROWMAX
				if !(Cabs1(a.Get(imax-1, imax-1)) < alpha*rowmax) {
					//                    interchange rows and columns K and IMAX,
					//                    use 1-by-1 pivot block
					kp = imax
					done = true

					//                 Equivalent to testing for ROWMAX .EQ. COLMAX,
					//                 used to handle NaN and Inf
				} else if (p == jmax) || (rowmax <= colmax) {
					//                    interchange rows and columns K+1 and IMAX,
					//                    use 2-by-2 pivot block
					kp = imax
					kstep = 2
					done = true
				} else {
					//                    Pivot NOT found, set variables and repeat
					p = imax
					colmax = rowmax
					imax = jmax
				}

				//                 End pivot search loop body
				if !done {
					goto label12
				}

			}

			//           Swap TWO rows and TWO columns
			//
			//           First swap
			if (kstep == 2) && (p != k) {
				//              Interchange rows and column K and P in the leading
				//              submatrix A(1:k,1:k) if we have a 2-by-2 pivot
				if p > 1 {
					goblas.Zswap(toPtr(p-1), a.CVector(0, k-1), func() *int { y := 1; return &y }(), a.CVector(0, p-1), func() *int { y := 1; return &y }())
				}
				if p < (k - 1) {
					goblas.Zswap(toPtr(k-p-1), a.CVector(p+1-1, k-1), func() *int { y := 1; return &y }(), a.CVector(p-1, p+1-1), lda)
				}
				t = a.Get(k-1, k-1)
				a.Set(k-1, k-1, a.Get(p-1, p-1))
				a.Set(p-1, p-1, t)

				//              Convert upper triangle of A into U form by applying
				//              the interchanges in columns k+1:N.
				if k < (*n) {
					goblas.Zswap(toPtr((*n)-k), a.CVector(k-1, k+1-1), lda, a.CVector(p-1, k+1-1), lda)
				}

			}

			//           Second swap
			kk = k - kstep + 1
			if kp != kk {
				//              Interchange rows and columns KK and KP in the leading
				//              submatrix A(1:k,1:k)
				if kp > 1 {
					goblas.Zswap(toPtr(kp-1), a.CVector(0, kk-1), func() *int { y := 1; return &y }(), a.CVector(0, kp-1), func() *int { y := 1; return &y }())
				}
				if (kk > 1) && (kp < (kk - 1)) {
					goblas.Zswap(toPtr(kk-kp-1), a.CVector(kp+1-1, kk-1), func() *int { y := 1; return &y }(), a.CVector(kp-1, kp+1-1), lda)
				}
				t = a.Get(kk-1, kk-1)
				a.Set(kk-1, kk-1, a.Get(kp-1, kp-1))
				a.Set(kp-1, kp-1, t)
				if kstep == 2 {
					t = a.Get(k-1-1, k-1)
					a.Set(k-1-1, k-1, a.Get(kp-1, k-1))
					a.Set(kp-1, k-1, t)
				}

				//              Convert upper triangle of A into U form by applying
				//              the interchanges in columns k+1:N.
				if k < (*n) {
					goblas.Zswap(toPtr((*n)-k), a.CVector(kk-1, k+1-1), lda, a.CVector(kp-1, k+1-1), lda)
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
					if Cabs1(a.Get(k-1, k-1)) >= sfmin {
						//                    Perform a rank-1 update of A(1:k-1,1:k-1) as
						//                    A := A - U(k)*D(k)*U(k)**T
						//                       = A - W(k)*1/D(k)*W(k)**T
						d11 = cone / a.Get(k-1, k-1)
						Zsyr(uplo, toPtr(k-1), toPtrc128(-d11), a.CVector(0, k-1), func() *int { y := 1; return &y }(), a, lda)

						//                    Store U(k) in column k
						goblas.Zscal(toPtr(k-1), &d11, a.CVector(0, k-1), func() *int { y := 1; return &y }())
					} else {
						//                    Store L(k) in column K
						d11 = a.Get(k-1, k-1)
						for ii = 1; ii <= k-1; ii++ {
							a.Set(ii-1, k-1, a.Get(ii-1, k-1)/d11)
						}

						//                    Perform a rank-1 update of A(k+1:n,k+1:n) as
						//                    A := A - U(k)*D(k)*U(k)**T
						//                       = A - W(k)*(1/D(k))*W(k)**T
						//                       = A - (W(k)/D(k))*(D(k))*(W(k)/D(K))**T
						Zsyr(uplo, toPtr(k-1), toPtrc128(-d11), a.CVector(0, k-1), func() *int { y := 1; return &y }(), a, lda)
					}

					//                 Store the superdiagonal element of D in array E
					e.Set(k-1, czero)

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

					d12 = a.Get(k-1-1, k-1)
					d22 = a.Get(k-1-1, k-1-1) / d12
					d11 = a.Get(k-1, k-1) / d12
					t = cone / (d11*d22 - cone)

					for j = k - 2; j >= 1; j-- {

						wkm1 = t * (d11*a.Get(j-1, k-1-1) - a.Get(j-1, k-1))
						wk = t * (d22*a.Get(j-1, k-1) - a.Get(j-1, k-1-1))
						//
						for i = j; i >= 1; i -= 1 {
							a.Set(i-1, j-1, a.Get(i-1, j-1)-(a.Get(i-1, k-1)/d12)*wk-(a.Get(i-1, k-1-1)/d12)*wkm1)
						}

						//                    Store U(k) and U(k-1) in cols k and k-1 for row J
						a.Set(j-1, k-1, wk/d12)
						a.Set(j-1, k-1-1, wkm1/d12)

					}

				}

				//              Copy superdiagonal elements of D(K) to E(K) and
				//              ZERO out superdiagonal entry of A
				e.Set(k-1, a.Get(k-1-1, k-1))
				e.Set(k-1-1, czero)
				a.Set(k-1-1, k-1, czero)

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

	label34:
	} else {
		//        Factorize A as L*D*L**T using the lower triangle of A
		//
		//        Initialize the unused last entry of the subdiagonal array E.
		e.Set((*n)-1, czero)

		//        K is the main loop index, increasing from 1 to N in steps of
		//        1 or 2
		k = 1
	label40:
		;

		//        If K > N, exit from loop
		if k > (*n) {
			goto label64
		}
		kstep = 1
		p = k

		//        Determine rows and columns to be interchanged and whether
		//        a 1-by-1 or 2-by-2 pivot block will be used
		absakk = Cabs1(a.Get(k-1, k-1))

		//        IMAX is the row-index of the largest off-diagonal element in
		//        column K, and COLMAX is its absolute value.
		//        Determine both COLMAX and IMAX.
		if k < (*n) {
			imax = k + goblas.Izamax(toPtr((*n)-k), a.CVector(k+1-1, k-1), func() *int { y := 1; return &y }())
			colmax = Cabs1(a.Get(imax-1, k-1))
		} else {
			colmax = zero
		}

		if maxf64(absakk, colmax) == zero {
			//           Column K is zero or underflow: set INFO and continue
			if (*info) == 0 {
				(*info) = k
			}
			kp = k

			//           Set E( K ) to zero
			if k < (*n) {
				e.Set(k-1, czero)
			}

		} else {
			//           Test for interchange
			//
			//           Equivalent to testing for (used to handle NaN and Inf)
			//           ABSAKK.GE.ALPHA*COLMAX
			if !(absakk < alpha*colmax) {
				//              no interchange, use 1-by-1 pivot block
				kp = k

			} else {

				done = false

				//              Loop until pivot found
			label42:
				;

				//                 Begin pivot search loop body
				//
				//                 JMAX is the column-index of the largest off-diagonal
				//                 element in row IMAX, and ROWMAX is its absolute value.
				//                 Determine both ROWMAX and JMAX.
				if imax != k {
					jmax = k - 1 + goblas.Izamax(toPtr(imax-k), a.CVector(imax-1, k-1), lda)
					rowmax = Cabs1(a.Get(imax-1, jmax-1))
				} else {
					rowmax = zero
				}

				if imax < (*n) {
					itemp = imax + goblas.Izamax(toPtr((*n)-imax), a.CVector(imax+1-1, imax-1), func() *int { y := 1; return &y }())
					dtemp = Cabs1(a.Get(itemp-1, imax-1))
					if dtemp > rowmax {
						rowmax = dtemp
						jmax = itemp
					}
				}

				//                 Equivalent to testing for (used to handle NaN and Inf)
				//                 ABS( A( IMAX, IMAX ) ).GE.ALPHA*ROWMAX
				if !(Cabs1(a.Get(imax-1, imax-1)) < alpha*rowmax) {
					//                    interchange rows and columns K and IMAX,
					//                    use 1-by-1 pivot block
					kp = imax
					done = true

					//                 Equivalent to testing for ROWMAX .EQ. COLMAX,
					//                 used to handle NaN and Inf
				} else if (p == jmax) || (rowmax <= colmax) {
					//                    interchange rows and columns K+1 and IMAX,
					//                    use 2-by-2 pivot block
					kp = imax
					kstep = 2
					done = true
				} else {
					//                    Pivot NOT found, set variables and repeat
					p = imax
					colmax = rowmax
					imax = jmax
				}

				//                 End pivot search loop body
				if !done {
					goto label42
				}

			}

			//           Swap TWO rows and TWO columns
			//
			//           First swap
			if (kstep == 2) && (p != k) {
				//              Interchange rows and column K and P in the trailing
				//              submatrix A(k:n,k:n) if we have a 2-by-2 pivot
				if p < (*n) {
					goblas.Zswap(toPtr((*n)-p), a.CVector(p+1-1, k-1), func() *int { y := 1; return &y }(), a.CVector(p+1-1, p-1), func() *int { y := 1; return &y }())
				}
				if p > (k + 1) {
					goblas.Zswap(toPtr(p-k-1), a.CVector(k+1-1, k-1), func() *int { y := 1; return &y }(), a.CVector(p-1, k+1-1), lda)
				}
				t = a.Get(k-1, k-1)
				a.Set(k-1, k-1, a.Get(p-1, p-1))
				a.Set(p-1, p-1, t)

				//              Convert lower triangle of A into L form by applying
				//              the interchanges in columns 1:k-1.
				if k > 1 {
					goblas.Zswap(toPtr(k-1), a.CVector(k-1, 0), lda, a.CVector(p-1, 0), lda)
				}

			}

			//           Second swap
			kk = k + kstep - 1
			if kp != kk {
				//              Interchange rows and columns KK and KP in the trailing
				//              submatrix A(k:n,k:n)
				if kp < (*n) {
					goblas.Zswap(toPtr((*n)-kp), a.CVector(kp+1-1, kk-1), func() *int { y := 1; return &y }(), a.CVector(kp+1-1, kp-1), func() *int { y := 1; return &y }())
				}
				if (kk < (*n)) && (kp > (kk + 1)) {
					goblas.Zswap(toPtr(kp-kk-1), a.CVector(kk+1-1, kk-1), func() *int { y := 1; return &y }(), a.CVector(kp-1, kk+1-1), lda)
				}
				t = a.Get(kk-1, kk-1)
				a.Set(kk-1, kk-1, a.Get(kp-1, kp-1))
				a.Set(kp-1, kp-1, t)
				if kstep == 2 {
					t = a.Get(k+1-1, k-1)
					a.Set(k+1-1, k-1, a.Get(kp-1, k-1))
					a.Set(kp-1, k-1, t)
				}

				//              Convert lower triangle of A into L form by applying
				//              the interchanges in columns 1:k-1.
				if k > 1 {
					goblas.Zswap(toPtr(k-1), a.CVector(kk-1, 0), lda, a.CVector(kp-1, 0), lda)
				}

			}

			//           Update the trailing submatrix
			if kstep == 1 {
				//              1-by-1 pivot block D(k): column k now holds
				//
				//              W(k) = L(k)*D(k)
				//
				//              where L(k) is the k-th column of L
				if k < (*n) {
					//              Perform a rank-1 update of A(k+1:n,k+1:n) and
					//              store L(k) in column k
					if Cabs1(a.Get(k-1, k-1)) >= sfmin {
						//                    Perform a rank-1 update of A(k+1:n,k+1:n) as
						//                    A := A - L(k)*D(k)*L(k)**T
						//                       = A - W(k)*(1/D(k))*W(k)**T
						d11 = cone / a.Get(k-1, k-1)
						Zsyr(uplo, toPtr((*n)-k), toPtrc128(-d11), a.CVector(k+1-1, k-1), func() *int { y := 1; return &y }(), a.Off(k+1-1, k+1-1), lda)

						//                    Store L(k) in column k
						goblas.Zscal(toPtr((*n)-k), &d11, a.CVector(k+1-1, k-1), func() *int { y := 1; return &y }())
					} else {
						//                    Store L(k) in column k
						d11 = a.Get(k-1, k-1)
						for ii = k + 1; ii <= (*n); ii++ {
							a.Set(ii-1, k-1, a.Get(ii-1, k-1)/d11)
						}

						//                    Perform a rank-1 update of A(k+1:n,k+1:n) as
						//                    A := A - L(k)*D(k)*L(k)**T
						//                       = A - W(k)*(1/D(k))*W(k)**T
						//                       = A - (W(k)/D(k))*(D(k))*(W(k)/D(K))**T
						Zsyr(uplo, toPtr((*n)-k), toPtrc128(-d11), a.CVector(k+1-1, k-1), func() *int { y := 1; return &y }(), a.Off(k+1-1, k+1-1), lda)
					}

					//                 Store the subdiagonal element of D in array E
					e.Set(k-1, czero)

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
				if k < (*n)-1 {

					d21 = a.Get(k+1-1, k-1)
					d11 = a.Get(k+1-1, k+1-1) / d21
					d22 = a.Get(k-1, k-1) / d21
					t = cone / (d11*d22 - cone)

					for j = k + 2; j <= (*n); j++ {
						//                    Compute  D21 * ( W(k)W(k+1) ) * inv(D(k)) for row J
						wk = t * (d11*a.Get(j-1, k-1) - a.Get(j-1, k+1-1))
						wkp1 = t * (d22*a.Get(j-1, k+1-1) - a.Get(j-1, k-1))

						//                    Perform a rank-2 update of A(k+2:n,k+2:n)
						for i = j; i <= (*n); i++ {
							a.Set(i-1, j-1, a.Get(i-1, j-1)-(a.Get(i-1, k-1)/d21)*wk-(a.Get(i-1, k+1-1)/d21)*wkp1)
						}

						//                    Store L(k) and L(k+1) in cols k and k+1 for row J
						a.Set(j-1, k-1, wk/d21)
						a.Set(j-1, k+1-1, wkp1/d21)

					}

				}

				//              Copy subdiagonal elements of D(K) to E(K) and
				//              ZERO out subdiagonal entry of A
				e.Set(k-1, a.Get(k+1-1, k-1))
				e.Set(k+1-1, czero)
				a.Set(k+1-1, k-1, czero)

			}

			//           End column K is nonsingular
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
		goto label40

	label64:
	}
}
