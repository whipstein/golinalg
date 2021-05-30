package golapack

import (
	"golinalg/goblas"
	"golinalg/golapack/gltest"
	"golinalg/mat"
	"math"
	"math/cmplx"
)

// Zhptrf computes the factorization of a complex Hermitian packed
// matrix A using the Bunch-Kaufman diagonal pivoting method:
//
//    A = U*D*U**H  or  A = L*D*L**H
//
// where U (or L) is a product of permutation and unit upper (lower)
// triangular matrices, and D is Hermitian and block diagonal with
// 1-by-1 and 2-by-2 diagonal blocks.
func Zhptrf(uplo byte, n *int, ap *mat.CVector, ipiv *[]int, info *int) {
	var upper bool
	var d12, d21, t, wk, wkm1, wkp1 complex128
	var absakk, alpha, colmax, d, d11, d22, eight, one, r1, rowmax, sevten, tt, zero float64
	var i, imax, j, jmax, k, kc, kk, knc, kp, kpc, kstep, kx, npp int

	zero = 0.0
	one = 1.0
	eight = 8.0
	sevten = 17.0

	Cabs1 := func(zdum complex128) float64 { return math.Abs(real(zdum)) + math.Abs(imag(zdum)) }

	//     Test the input parameters.
	(*info) = 0
	upper = uplo == 'U'
	if !upper && uplo != 'L' {
		(*info) = -1
	} else if (*n) < 0 {
		(*info) = -2
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("ZHPTRF"), -(*info))
		return
	}

	//     Initialize ALPHA for use in choosing pivot block size.
	alpha = (one + math.Sqrt(sevten)) / eight

	if upper {
		//        Factorize A as U*D*U**H using the upper triangle of A
		//
		//        K is the main loop index, decreasing from N to 1 in steps of
		//        1 or 2
		k = (*n)
		kc = ((*n)-1)*(*n)/2 + 1
	label10:
		;
		knc = kc

		//        If K < 1, exit from loop
		if k < 1 {
			return
		}
		kstep = 1

		//        Determine rows and columns to be interchanged and whether
		//        a 1-by-1 or 2-by-2 pivot block will be used
		absakk = math.Abs(ap.GetRe(kc + k - 1 - 1))

		//        IMAX is the row-index of the largest off-diagonal element in
		//        column K, and COLMAX is its absolute value
		if k > 1 {
			imax = goblas.Izamax(toPtr(k-1), ap.Off(kc-1), func() *int { y := 1; return &y }())
			colmax = Cabs1(ap.Get(kc + imax - 1 - 1))
		} else {
			colmax = zero
		}

		if maxf64(absakk, colmax) == zero {
			//           Column K is zero: set INFO and continue
			if (*info) == 0 {
				(*info) = k
			}
			kp = k
			ap.Set(kc+k-1-1, ap.GetReCmplx(kc+k-1-1))
		} else {
			if absakk >= alpha*colmax {
				//              no interchange, use 1-by-1 pivot block
				kp = k
			} else {
				//              JMAX is the column-index of the largest off-diagonal
				//              element in row IMAX, and ROWMAX is its absolute value
				rowmax = zero
				jmax = imax
				kx = imax*(imax+1)/2 + imax
				for j = imax + 1; j <= k; j++ {
					if Cabs1(ap.Get(kx-1)) > rowmax {
						rowmax = Cabs1(ap.Get(kx - 1))
						jmax = j
					}
					kx = kx + j
				}
				kpc = (imax-1)*imax/2 + 1
				if imax > 1 {
					jmax = goblas.Izamax(toPtr(imax-1), ap.Off(kpc-1), func() *int { y := 1; return &y }())
					rowmax = maxf64(rowmax, Cabs1(ap.Get(kpc+jmax-1-1)))
				}

				if absakk >= alpha*colmax*(colmax/rowmax) {
					//                 no interchange, use 1-by-1 pivot block
					kp = k
				} else if math.Abs(ap.GetRe(kpc+imax-1-1)) >= alpha*rowmax {
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
			if kstep == 2 {
				knc = knc - k + 1
			}
			if kp != kk {
				//              Interchange rows and columns KK and KP in the leading
				//              submatrix A(1:k,1:k)
				goblas.Zswap(toPtr(kp-1), ap.Off(knc-1), func() *int { y := 1; return &y }(), ap.Off(kpc-1), func() *int { y := 1; return &y }())
				kx = kpc + kp - 1
				for j = kp + 1; j <= kk-1; j++ {
					kx = kx + j - 1
					t = ap.GetConj(knc + j - 1 - 1)
					ap.Set(knc+j-1-1, ap.GetConj(kx-1))
					ap.Set(kx-1, t)
				}
				ap.Set(kx+kk-1-1, ap.GetConj(kx+kk-1-1))
				r1 = ap.GetRe(knc + kk - 1 - 1)
				ap.Set(knc+kk-1-1, ap.GetReCmplx(kpc+kp-1-1))
				ap.SetRe(kpc+kp-1-1, r1)
				if kstep == 2 {
					ap.Set(kc+k-1-1, ap.GetReCmplx(kc+k-1-1))
					t = ap.Get(kc + k - 2 - 1)
					ap.Set(kc+k-2-1, ap.Get(kc+kp-1-1))
					ap.Set(kc+kp-1-1, t)
				}
			} else {
				ap.Set(kc+k-1-1, ap.GetReCmplx(kc+k-1-1))
				if kstep == 2 {
					ap.Set(kc-1-1, ap.GetReCmplx(kc-1-1))
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
				//              A := A - U(k)*D(k)*U(k)**H = A - W(k)*1/D(k)*W(k)**H
				r1 = one / ap.GetRe(kc+k-1-1)
				goblas.Zhpr(mat.UploByte(uplo), toPtr(k-1), toPtrf64(-r1), ap.Off(kc-1), func() *int { y := 1; return &y }(), ap)

				//              Store U(k) in column k
				goblas.Zdscal(toPtr(k-1), &r1, ap.Off(kc-1), func() *int { y := 1; return &y }())
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
				//              A := A - ( U(k-1) U(k) )*D(k)*( U(k-1) U(k) )**H
				//                 = A - ( W(k-1) W(k) )*inv(D(k))*( W(k-1) W(k) )**H
				if k > 2 {

					d = Dlapy2(toPtrf64(ap.GetRe(k-1+(k-1)*k/2-1)), toPtrf64(ap.GetIm(k-1+(k-1)*k/2-1)))
					d22 = ap.GetRe(k-1+(k-2)*(k-1)/2-1) / d
					d11 = ap.GetRe(k+(k-1)*k/2-1) / d
					tt = one / (d11*d22 - one)
					d12 = ap.Get(k-1+(k-1)*k/2-1) / complex(d, 0)
					d = tt / d

					for j = k - 2; j >= 1; j-- {
						wkm1 = complex(d, 0) * (complex(d11, 0)*ap.Get(j+(k-2)*(k-1)/2-1) - cmplx.Conj(d12)*ap.Get(j+(k-1)*k/2-1))
						wk = complex(d, 0) * (complex(d22, 0)*ap.Get(j+(k-1)*k/2-1) - d12*ap.Get(j+(k-2)*(k-1)/2-1))
						for i = j; i >= 1; i-- {
							ap.Set(i+(j-1)*j/2-1, ap.Get(i+(j-1)*j/2-1)-ap.Get(i+(k-1)*k/2-1)*cmplx.Conj(wk)-ap.Get(i+(k-2)*(k-1)/2-1)*cmplx.Conj(wkm1))
						}
						ap.Set(j+(k-1)*k/2-1, wk)
						ap.Set(j+(k-2)*(k-1)/2-1, wkm1)
						ap.Set(j+(j-1)*j/2-1, ap.GetReCmplx(j+(j-1)*j/2-1))
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
		kc = knc - k
		goto label10

	} else {
		//        Factorize A as L*D*L**H using the lower triangle of A
		//
		//        K is the main loop index, increasing from 1 to N in steps of
		//        1 or 2
		k = 1
		kc = 1
		npp = (*n) * ((*n) + 1) / 2
	label60:
		;
		knc = kc

		//        If K > N, exit from loop
		if k > (*n) {
			return
		}
		kstep = 1

		//        Determine rows and columns to be interchanged and whether
		//        a 1-by-1 or 2-by-2 pivot block will be used
		absakk = math.Abs(ap.GetRe(kc - 1))

		//        IMAX is the row-index of the largest off-diagonal element in
		//        column K, and COLMAX is its absolute value
		if k < (*n) {
			imax = k + goblas.Izamax(toPtr((*n)-k), ap.Off(kc+1-1), func() *int { y := 1; return &y }())
			colmax = Cabs1(ap.Get(kc + imax - k - 1))
		} else {
			colmax = zero
		}

		if maxf64(absakk, colmax) == zero {
			//           Column K is zero: set INFO and continue
			if (*info) == 0 {
				(*info) = k
			}
			kp = k
			ap.Set(kc-1, ap.GetReCmplx(kc-1))
		} else {
			if absakk >= alpha*colmax {
				//              no interchange, use 1-by-1 pivot block
				kp = k
			} else {
				//              JMAX is the column-index of the largest off-diagonal
				//              element in row IMAX, and ROWMAX is its absolute value
				rowmax = zero
				kx = kc + imax - k
				for j = k; j <= imax-1; j++ {
					if Cabs1(ap.Get(kx-1)) > rowmax {
						rowmax = Cabs1(ap.Get(kx - 1))
						jmax = j
					}
					kx = kx + (*n) - j
				}
				kpc = npp - ((*n)-imax+1)*((*n)-imax+2)/2 + 1
				if imax < (*n) {
					jmax = imax + goblas.Izamax(toPtr((*n)-imax), ap.Off(kpc+1-1), func() *int { y := 1; return &y }())
					rowmax = maxf64(rowmax, Cabs1(ap.Get(kpc+jmax-imax-1)))
				}

				if absakk >= alpha*colmax*(colmax/rowmax) {
					//                 no interchange, use 1-by-1 pivot block
					kp = k
				} else if math.Abs(ap.GetRe(kpc-1)) >= alpha*rowmax {
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
			if kstep == 2 {
				knc = knc + (*n) - k + 1
			}
			if kp != kk {
				//              Interchange rows and columns KK and KP in the trailing
				//              submatrix A(k:n,k:n)
				if kp < (*n) {
					goblas.Zswap(toPtr((*n)-kp), ap.Off(knc+kp-kk+1-1), func() *int { y := 1; return &y }(), ap.Off(kpc+1-1), func() *int { y := 1; return &y }())
				}
				kx = knc + kp - kk
				for j = kk + 1; j <= kp-1; j++ {
					kx = kx + (*n) - j + 1
					t = cmplx.Conj(ap.Get(knc + j - kk - 1))
					ap.Set(knc+j-kk-1, cmplx.Conj(ap.Get(kx-1)))
					ap.Set(kx-1, t)
				}
				ap.Set(knc+kp-kk-1, cmplx.Conj(ap.Get(knc+kp-kk-1)))
				r1 = ap.GetRe(knc - 1)
				ap.Set(knc-1, ap.GetReCmplx(kpc-1))
				ap.SetRe(kpc-1, r1)
				if kstep == 2 {
					ap.Set(kc-1, ap.GetReCmplx(kc-1))
					t = ap.Get(kc + 1 - 1)
					ap.Set(kc+1-1, ap.Get(kc+kp-k-1))
					ap.Set(kc+kp-k-1, t)
				}
			} else {
				ap.Set(kc-1, ap.GetReCmplx(kc-1))
				if kstep == 2 {
					ap.Set(knc-1, ap.GetReCmplx(knc-1))
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
					//                 Perform a rank-1 update of A(k+1:n,k+1:n) as
					//
					//                 A := A - L(k)*D(k)*L(k)**H = A - W(k)*(1/D(k))*W(k)**H
					r1 = one / ap.GetRe(kc-1)
					goblas.Zhpr(mat.UploByte(uplo), toPtr((*n)-k), toPtrf64(-r1), ap.Off(kc+1-1), func() *int { y := 1; return &y }(), ap.Off(kc+(*n)-k+1-1))

					//                 Store L(k) in column K
					goblas.Zdscal(toPtr((*n)-k), &r1, ap.Off(kc+1-1), func() *int { y := 1; return &y }())
				}
			} else {
				//              2-by-2 pivot block D(k): columns K and K+1 now hold
				//
				//              ( W(k) W(k+1) ) = ( L(k) L(k+1) )*D(k)
				//
				//              where L(k) and L(k+1) are the k-th and (k+1)-th columns
				//              of L
				if k < (*n)-1 {
					//                 Perform a rank-2 update of A(k+2:n,k+2:n) as
					//
					//                 A := A - ( L(k) L(k+1) )*D(k)*( L(k) L(k+1) )**H
					//                    = A - ( W(k) W(k+1) )*inv(D(k))*( W(k) W(k+1) )**H
					//
					//                 where L(k) and L(k+1) are the k-th and (k+1)-th
					//                 columns of L
					d = Dlapy2(toPtrf64(ap.GetRe(k+1+(k-1)*(2*(*n)-k)/2-1)), toPtrf64(ap.GetIm(k+1+(k-1)*(2*(*n)-k)/2-1)))
					d11 = ap.GetRe(k+1+k*(2*(*n)-k-1)/2-1) / d
					d22 = ap.GetRe(k+(k-1)*(2*(*n)-k)/2-1) / d
					tt = one / (d11*d22 - one)
					d21 = ap.Get(k+1+(k-1)*(2*(*n)-k)/2-1) / complex(d, 0)
					d = tt / d

					for j = k + 2; j <= (*n); j++ {
						wk = complex(d, 0) * (complex(d11, 0)*ap.Get(j+(k-1)*(2*(*n)-k)/2-1) - d21*ap.Get(j+k*(2*(*n)-k-1)/2-1))
						wkp1 = complex(d, 0) * (complex(d22, 0)*ap.Get(j+k*(2*(*n)-k-1)/2-1) - cmplx.Conj(d21)*ap.Get(j+(k-1)*(2*(*n)-k)/2-1))
						for i = j; i <= (*n); i++ {
							ap.Set(i+(j-1)*(2*(*n)-j)/2-1, ap.Get(i+(j-1)*(2*(*n)-j)/2-1)-ap.Get(i+(k-1)*(2*(*n)-k)/2-1)*cmplx.Conj(wk)-ap.Get(i+k*(2*(*n)-k-1)/2-1)*cmplx.Conj(wkp1))
						}
						ap.Set(j+(k-1)*(2*(*n)-k)/2-1, wk)
						ap.Set(j+k*(2*(*n)-k-1)/2-1, wkp1)
						ap.Set(j+(j-1)*(2*(*n)-j)/2-1, ap.GetReCmplx(j+(j-1)*(2*(*n)-j)/2-1))
					}
				}
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
		kc = knc + (*n) - k + 2
		goto label60

	}
}
