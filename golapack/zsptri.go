package golapack

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zsptri computes the inverse of a complex symmetric indefinite matrix
// A in packed storage using the factorization A = U*D*U**T or
// A = L*D*L**T computed by ZSPTRF.
func Zsptri(uplo byte, n *int, ap *mat.CVector, ipiv *[]int, work *mat.CVector, info *int) {
	var upper bool
	var ak, akkp1, akp1, d, one, t, temp, zero complex128
	var j, k, kc, kcnext, kp, kpc, kstep, kx, npp int

	one = (1.0 + 0.0*1i)
	zero = (0.0 + 0.0*1i)

	//     Test the input parameters.
	(*info) = 0
	upper = uplo == 'U'
	if !upper && uplo != 'L' {
		(*info) = -1
	} else if (*n) < 0 {
		(*info) = -2
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("ZSPTRI"), -(*info))
		return
	}

	//     Quick return if possible
	if (*n) == 0 {
		return
	}

	//     Check that the diagonal matrix D is nonsingular.
	if upper {
		//        Upper triangular storage: examine D from bottom to top
		kp = (*n) * ((*n) + 1) / 2
		for (*info) = (*n); (*info) >= 1; (*info)-- {
			if (*ipiv)[(*info)-1] > 0 && ap.Get(kp-1) == zero {
				return
			}
			kp = kp - (*info)
		}
	} else {
		//        Lower triangular storage: examine D from top to bottom.
		kp = 1
		for (*info) = 1; (*info) <= (*n); (*info)++ {
			if (*ipiv)[(*info)-1] > 0 && ap.Get(kp-1) == zero {
				return
			}
			kp = kp + (*n) - (*info) + 1
		}
	}
	(*info) = 0

	if upper {
		//        Compute inv(A) from the factorization A = U*D*U**T.
		//
		//        K is the main loop index, increasing from 1 to N in steps of
		//        1 or 2, depending on the size of the diagonal blocks.
		k = 1
		kc = 1
	label30:
		;

		//        If K > N, exit from loop.
		if k > (*n) {
			goto label50
		}

		kcnext = kc + k
		if (*ipiv)[k-1] > 0 {
			//           1 x 1 diagonal block
			//
			//           Invert the diagonal block.
			ap.Set(kc+k-1-1, one/ap.Get(kc+k-1-1))

			//           Compute column K of the inverse.
			if k > 1 {
				goblas.Zcopy(k-1, ap.Off(kc-1, 1), work.Off(0, 1))
				Zspmv(uplo, toPtr(k-1), toPtrc128(-one), ap, work, func() *int { y := 1; return &y }(), &zero, ap.Off(kc-1), func() *int { y := 1; return &y }())
				ap.Set(kc+k-1-1, ap.Get(kc+k-1-1)-goblas.Zdotu(k-1, work.Off(0, 1), ap.Off(kc-1, 1)))
			}
			kstep = 1
		} else {
			//           2 x 2 diagonal block
			//
			//           Invert the diagonal block.
			t = ap.Get(kcnext + k - 1 - 1)
			ak = ap.Get(kc+k-1-1) / t
			akp1 = ap.Get(kcnext+k-1) / t
			akkp1 = ap.Get(kcnext+k-1-1) / t
			d = t * (ak*akp1 - one)
			ap.Set(kc+k-1-1, akp1/d)
			ap.Set(kcnext+k-1, ak/d)
			ap.Set(kcnext+k-1-1, -akkp1/d)

			//           Compute columns K and K+1 of the inverse.
			if k > 1 {
				goblas.Zcopy(k-1, ap.Off(kc-1, 1), work.Off(0, 1))
				Zspmv(uplo, toPtr(k-1), toPtrc128(-one), ap, work, func() *int { y := 1; return &y }(), &zero, ap.Off(kc-1), func() *int { y := 1; return &y }())
				ap.Set(kc+k-1-1, ap.Get(kc+k-1-1)-goblas.Zdotu(k-1, work.Off(0, 1), ap.Off(kc-1, 1)))
				ap.Set(kcnext+k-1-1, ap.Get(kcnext+k-1-1)-goblas.Zdotu(k-1, ap.Off(kc-1, 1), ap.Off(kcnext-1, 1)))
				goblas.Zcopy(k-1, ap.Off(kcnext-1, 1), work.Off(0, 1))
				Zspmv(uplo, toPtr(k-1), toPtrc128(-one), ap, work, func() *int { y := 1; return &y }(), &zero, ap.Off(kcnext-1), func() *int { y := 1; return &y }())
				ap.Set(kcnext+k-1, ap.Get(kcnext+k-1)-goblas.Zdotu(k-1, work.Off(0, 1), ap.Off(kcnext-1, 1)))
			}
			kstep = 2
			kcnext = kcnext + k + 1
		}

		kp = abs((*ipiv)[k-1])
		if kp != k {
			//           Interchange rows and columns K and KP in the leading
			//           submatrix A(1:k+1,1:k+1)
			kpc = (kp-1)*kp/2 + 1
			goblas.Zswap(kp-1, ap.Off(kc-1, 1), ap.Off(kpc-1, 1))
			kx = kpc + kp - 1
			for j = kp + 1; j <= k-1; j++ {
				kx = kx + j - 1
				temp = ap.Get(kc + j - 1 - 1)
				ap.Set(kc+j-1-1, ap.Get(kx-1))
				ap.Set(kx-1, temp)
			}
			temp = ap.Get(kc + k - 1 - 1)
			ap.Set(kc+k-1-1, ap.Get(kpc+kp-1-1))
			ap.Set(kpc+kp-1-1, temp)
			if kstep == 2 {
				temp = ap.Get(kc + k + k - 1 - 1)
				ap.Set(kc+k+k-1-1, ap.Get(kc+k+kp-1-1))
				ap.Set(kc+k+kp-1-1, temp)
			}
		}

		k = k + kstep
		kc = kcnext
		goto label30
	label50:
	} else {
		//
		//        Compute inv(A) from the factorization A = L*D*L**T.
		//
		//        K is the main loop index, increasing from 1 to N in steps of
		//        1 or 2, depending on the size of the diagonal blocks.
		npp = (*n) * ((*n) + 1) / 2
		k = (*n)
		kc = npp
	label60:
		;

		//        If K < 1, exit from loop.
		if k < 1 {
			goto label80
		}

		kcnext = kc - ((*n) - k + 2)
		if (*ipiv)[k-1] > 0 {
			//           1 x 1 diagonal block
			//
			//           Invert the diagonal block.
			ap.Set(kc-1, one/ap.Get(kc-1))

			//           Compute column K of the inverse.
			if k < (*n) {
				goblas.Zcopy((*n)-k, ap.Off(kc, 1), work.Off(0, 1))
				Zspmv(uplo, toPtr((*n)-k), toPtrc128(-one), ap.Off(kc+(*n)-k), work, func() *int { y := 1; return &y }(), &zero, ap.Off(kc), func() *int { y := 1; return &y }())
				ap.Set(kc-1, ap.Get(kc-1)-goblas.Zdotu((*n)-k, work.Off(0, 1), ap.Off(kc, 1)))
			}
			kstep = 1
		} else {
			//           2 x 2 diagonal block
			//
			//           Invert the diagonal block.
			t = ap.Get(kcnext + 1 - 1)
			ak = ap.Get(kcnext-1) / t
			akp1 = ap.Get(kc-1) / t
			akkp1 = ap.Get(kcnext) / t
			d = t * (ak*akp1 - one)
			ap.Set(kcnext-1, akp1/d)
			ap.Set(kc-1, ak/d)
			ap.Set(kcnext, -akkp1/d)
			//
			//           Compute columns K-1 and K of the inverse.
			if k < (*n) {
				goblas.Zcopy((*n)-k, ap.Off(kc, 1), work.Off(0, 1))
				Zspmv(uplo, toPtr((*n)-k), toPtrc128(-one), ap.Off(kc+((*n)-k+1)-1), work, func() *int { y := 1; return &y }(), &zero, ap.Off(kc), func() *int { y := 1; return &y }())
				ap.Set(kc-1, ap.Get(kc-1)-goblas.Zdotu((*n)-k, work.Off(0, 1), ap.Off(kc, 1)))
				ap.Set(kcnext, ap.Get(kcnext)-goblas.Zdotu((*n)-k, ap.Off(kc, 1), ap.Off(kcnext+2-1, 1)))
				goblas.Zcopy((*n)-k, ap.Off(kcnext+2-1, 1), work.Off(0, 1))
				Zspmv(uplo, toPtr((*n)-k), toPtrc128(-one), ap.Off(kc+((*n)-k+1)-1), work, func() *int { y := 1; return &y }(), &zero, ap.Off(kcnext+2-1), func() *int { y := 1; return &y }())
				ap.Set(kcnext-1, ap.Get(kcnext-1)-goblas.Zdotu((*n)-k, work.Off(0, 1), ap.Off(kcnext+2-1, 1)))
			}
			kstep = 2
			kcnext = kcnext - ((*n) - k + 3)
		}

		kp = abs((*ipiv)[k-1])
		if kp != k {
			//           Interchange rows and columns K and KP in the trailing
			//           submatrix A(k-1:n,k-1:n)
			kpc = npp - ((*n)-kp+1)*((*n)-kp+2)/2 + 1
			if kp < (*n) {
				goblas.Zswap((*n)-kp, ap.Off(kc+kp-k, 1), ap.Off(kpc, 1))
			}
			kx = kc + kp - k
			for j = k + 1; j <= kp-1; j++ {
				kx = kx + (*n) - j + 1
				temp = ap.Get(kc + j - k - 1)
				ap.Set(kc+j-k-1, ap.Get(kx-1))
				ap.Set(kx-1, temp)
			}
			temp = ap.Get(kc - 1)
			ap.Set(kc-1, ap.Get(kpc-1))
			ap.Set(kpc-1, temp)
			if kstep == 2 {
				temp = ap.Get(kc - (*n) + k - 1 - 1)
				ap.Set(kc-(*n)+k-1-1, ap.Get(kc-(*n)+kp-1-1))
				ap.Set(kc-(*n)+kp-1-1, temp)
			}
		}

		k = k - kstep
		kc = kcnext
		goto label60
	label80:
	}
}
