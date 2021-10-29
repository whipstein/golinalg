package golapack

import (
	"fmt"
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zsytf2 computes the factorization of a complex symmetric matrix A
// using the Bunch-Kaufman diagonal pivoting method:
//
//    A = U*D*U**T  or  A = L*D*L**T
//
// where U (or L) is a product of permutation and unit upper (lower)
// triangular matrices, U**T is the transpose of U, and D is symmetric and
// block diagonal with 1-by-1 and 2-by-2 diagonal blocks.
//
// This is the unblocked version of the algorithm, calling Level 2 BLAS.
func Zsytf2(uplo mat.MatUplo, n int, a *mat.CMatrix, ipiv *[]int) (info int, err error) {
	var upper bool
	var cone, d11, d12, d21, d22, r1, t, wk, wkm1, wkp1 complex128
	var absakk, alpha, colmax, eight, one, rowmax, sevten, zero float64
	var i, imax, j, jmax, k, kk, kp, kstep int

	zero = 0.0
	one = 1.0
	eight = 8.0
	sevten = 17.0
	cone = (1.0 + 0.0*1i)

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
		gltest.Xerbla2("Zsytf2", err)
		return
	}

	//     Initialize ALPHA for use in choosing pivot block size.
	alpha = (one + math.Sqrt(sevten)) / eight

	if upper {
		//        Factorize A as U*D*U**T using the upper triangle of A
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

		//        Determine rows and columns to be interchanged and whether
		//        a 1-by-1 or 2-by-2 pivot block will be used
		absakk = cabs1(a.Get(k-1, k-1))

		//        IMAX is the row-index of the largest off-diagonal element in
		//        column K, and COLMAX is its absolute value.
		//        Determine both COLMAX and IMAX.
		if k > 1 {
			imax = goblas.Izamax(k-1, a.CVector(0, k-1, 1))
			colmax = cabs1(a.Get(imax-1, k-1))
		} else {
			colmax = zero
		}

		if math.Max(absakk, colmax) == zero || Disnan(int(absakk)) {
			//           Column K is zero or underflow, or contains a NaN:
			//           set INFO and continue
			if info == 0 {
				info = k
			}
			kp = k
		} else {
			if absakk >= alpha*colmax {
				//              no interchange, use 1-by-1 pivot block
				kp = k
			} else {
				//              JMAX is the column-index of the largest off-diagonal
				//              element in row IMAX, and ROWMAX is its absolute value
				jmax = imax + goblas.Izamax(k-imax, a.CVector(imax-1, imax))
				rowmax = cabs1(a.Get(imax-1, jmax-1))
				if imax > 1 {
					jmax = goblas.Izamax(imax-1, a.CVector(0, imax-1, 1))
					rowmax = math.Max(rowmax, cabs1(a.Get(jmax-1, imax-1)))
				}

				if absakk >= alpha*colmax*(colmax/rowmax) {
					//                 no interchange, use 1-by-1 pivot block
					kp = k
				} else if cabs1(a.Get(imax-1, imax-1)) >= alpha*rowmax {
					//                 interchange rows and columns K and IMAX, use 1-by-1
					//                 pivot block
					kp = imax
				} else {
					//                 interchange rows and columns K-1 and IMAX, use 2-by-2
					//                 pivot block
					kp = imax
					kstep = 2
				}
			}

			kk = k - kstep + 1
			if kp != kk {
				//              Interchange rows and columns KK and KP in the leading
				//              submatrix A(1:k,1:k)
				goblas.Zswap(kp-1, a.CVector(0, kk-1, 1), a.CVector(0, kp-1, 1))
				goblas.Zswap(kk-kp-1, a.CVector(kp, kk-1, 1), a.CVector(kp-1, kp))
				t = a.Get(kk-1, kk-1)
				a.Set(kk-1, kk-1, a.Get(kp-1, kp-1))
				a.Set(kp-1, kp-1, t)
				if kstep == 2 {
					t = a.Get(k-1-1, k-1)
					a.Set(k-1-1, k-1, a.Get(kp-1, k-1))
					a.Set(kp-1, k-1, t)
				}
			}

			//           Update the leading submatrix
			if kstep == 1 {
				//              1-by-1 pivot block D(k): column k now holds
				//
				//              W(k) = U(k)*D(k)
				//
				//              where U(k) is the k-th column of U
				//
				//              Perform a rank-1 update of A(1:k-1,1:k-1) as
				//
				//              A := A - U(k)*D(k)*U(k)**T = A - W(k)*1/D(k)*W(k)**T
				r1 = cone / a.Get(k-1, k-1)
				if err = Zsyr(uplo, k-1, -r1, a.CVector(0, k-1, 1), a); err != nil {
					panic(err)
				}

				//              Store U(k) in column k
				goblas.Zscal(k-1, r1, a.CVector(0, k-1, 1))
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
				//                 = A - ( W(k-1) W(k) )*inv(D(k))*( W(k-1) W(k) )**T
				if k > 2 {

					d12 = a.Get(k-1-1, k-1)
					d22 = a.Get(k-1-1, k-1-1) / d12
					d11 = a.Get(k-1, k-1) / d12
					t = cone / (d11*d22 - cone)
					d12 = t / d12

					for j = k - 2; j >= 1; j-- {
						wkm1 = d12 * (d11*a.Get(j-1, k-1-1) - a.Get(j-1, k-1))
						wk = d12 * (d22*a.Get(j-1, k-1) - a.Get(j-1, k-1-1))
						for i = j; i >= 1; i-- {
							a.Set(i-1, j-1, a.Get(i-1, j-1)-a.Get(i-1, k-1)*wk-a.Get(i-1, k-1-1)*wkm1)
						}
						a.Set(j-1, k-1, wk)
						a.Set(j-1, k-1-1, wkm1)
					}

				}

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

	} else {
		//        Factorize A as L*D*L**T using the lower triangle of A
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

		//        Determine rows and columns to be interchanged and whether
		//        a 1-by-1 or 2-by-2 pivot block will be used
		absakk = cabs1(a.Get(k-1, k-1))

		//        IMAX is the row-index of the largest off-diagonal element in
		//        column K, and COLMAX is its absolute value.
		//        Determine both COLMAX and IMAX.
		if k < n {
			imax = k + goblas.Izamax(n-k, a.CVector(k, k-1, 1))
			colmax = cabs1(a.Get(imax-1, k-1))
		} else {
			colmax = zero
		}

		if math.Max(absakk, colmax) == zero || Disnan(int(absakk)) {
			//           Column K is zero or underflow, or contains a NaN:
			//           set INFO and continue
			if info == 0 {
				info = k
			}
			kp = k
		} else {
			if absakk >= alpha*colmax {
				//              no interchange, use 1-by-1 pivot block
				kp = k
			} else {
				//              JMAX is the column-index of the largest off-diagonal
				//              element in row IMAX, and ROWMAX is its absolute value
				jmax = k - 1 + goblas.Izamax(imax-k, a.CVector(imax-1, k-1))
				rowmax = cabs1(a.Get(imax-1, jmax-1))
				if imax < n {
					jmax = imax + goblas.Izamax(n-imax, a.CVector(imax, imax-1, 1))
					rowmax = math.Max(rowmax, cabs1(a.Get(jmax-1, imax-1)))
				}

				if absakk >= alpha*colmax*(colmax/rowmax) {
					//                 no interchange, use 1-by-1 pivot block
					kp = k
				} else if cabs1(a.Get(imax-1, imax-1)) >= alpha*rowmax {
					//                 interchange rows and columns K and IMAX, use 1-by-1
					//                 pivot block
					kp = imax
				} else {
					//                 interchange rows and columns K+1 and IMAX, use 2-by-2
					//                 pivot block
					kp = imax
					kstep = 2
				}
			}

			kk = k + kstep - 1
			if kp != kk {
				//              Interchange rows and columns KK and KP in the trailing
				//              submatrix A(k:n,k:n)
				if kp < n {
					goblas.Zswap(n-kp, a.CVector(kp, kk-1, 1), a.CVector(kp, kp-1, 1))
				}
				goblas.Zswap(kp-kk-1, a.CVector(kk, kk-1, 1), a.CVector(kp-1, kk))
				t = a.Get(kk-1, kk-1)
				a.Set(kk-1, kk-1, a.Get(kp-1, kp-1))
				a.Set(kp-1, kp-1, t)
				if kstep == 2 {
					t = a.Get(k, k-1)
					a.Set(k, k-1, a.Get(kp-1, k-1))
					a.Set(kp-1, k-1, t)
				}
			}

			//           Update the trailing submatrix
			if kstep == 1 {
				//              1-by-1 pivot block D(k): column k now holds
				//
				//              W(k) = L(k)*D(k)
				//
				//              where L(k) is the k-th column of L
				if k < n {
					//                 Perform a rank-1 update of A(k+1:n,k+1:n) as
					//
					//                 A := A - L(k)*D(k)*L(k)**T = A - W(k)*(1/D(k))*W(k)**T
					r1 = cone / a.Get(k-1, k-1)
					if err = Zsyr(uplo, n-k, -r1, a.CVector(k, k-1, 1), a.Off(k, k)); err != nil {
						panic(err)
					}

					//                 Store L(k) in column K
					goblas.Zscal(n-k, r1, a.CVector(k, k-1, 1))
				}
			} else {
				//              2-by-2 pivot block D(k)
				if k < n-1 {
					//                 Perform a rank-2 update of A(k+2:n,k+2:n) as
					//
					//                 A := A - ( L(k) L(k+1) )*D(k)*( L(k) L(k+1) )**T
					//                    = A - ( W(k) W(k+1) )*inv(D(k))*( W(k) W(k+1) )**T
					//
					//                 where L(k) and L(k+1) are the k-th and (k+1)-th
					//                 columns of L
					d21 = a.Get(k, k-1)
					d11 = a.Get(k, k) / d21
					d22 = a.Get(k-1, k-1) / d21
					t = cone / (d11*d22 - cone)
					d21 = t / d21

					for j = k + 2; j <= n; j++ {
						wk = d21 * (d11*a.Get(j-1, k-1) - a.Get(j-1, k))
						wkp1 = d21 * (d22*a.Get(j-1, k) - a.Get(j-1, k-1))
						for i = j; i <= n; i++ {
							a.Set(i-1, j-1, a.Get(i-1, j-1)-a.Get(i-1, k-1)*wk-a.Get(i-1, k)*wkp1)
						}
						a.Set(j-1, k-1, wk)
						a.Set(j-1, k, wkp1)
					}
				}
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
		goto label40

	}
}
