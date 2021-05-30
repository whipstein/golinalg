package golapack

import (
	"golinalg/goblas"
	"golinalg/golapack/gltest"
	"golinalg/mat"
)

// Zsytrirook computes the inverse of a complex symmetric
// matrix A using the factorization A = U*D*U**T or A = L*D*L**T
// computed by ZSYTRF_ROOK.
func Zsytrirook(uplo byte, n *int, a *mat.CMatrix, lda *int, ipiv *[]int, work *mat.CVector, info *int) {
	var upper bool
	var ak, akkp1, akp1, cone, czero, d, t, temp complex128
	var k, kp, kstep int

	cone = (1.0 + 0.0*1i)
	czero = (0.0 + 0.0*1i)

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
		gltest.Xerbla([]byte("ZSYTRI_ROOK"), -(*info))
		return
	}

	//     Quick return if possible
	if (*n) == 0 {
		return
	}

	//     Check that the diagonal matrix D is nonsingular.
	if upper {
		//        Upper triangular storage: examine D from bottom to top
		for (*info) = (*n); (*info) >= 1; (*info)-- {
			if (*ipiv)[(*info)-1] > 0 && a.Get((*info)-1, (*info)-1) == czero {
				return
			}
		}
	} else {
		//        Lower triangular storage: examine D from top to bottom.
		for (*info) = 1; (*info) <= (*n); (*info)++ {
			if (*ipiv)[(*info)-1] > 0 && a.Get((*info)-1, (*info)-1) == czero {
				return
			}
		}
	}
	(*info) = 0

	if upper {
		//        Compute inv(A) from the factorization A = U*D*U**T.
		//
		//        K is the main loop index, increasing from 1 to N in steps of
		//        1 or 2, depending on the size of the diagonal blocks.
		k = 1
	label30:
		;

		//        If K > N, exit from loop.
		if k > (*n) {
			goto label40
		}

		if (*ipiv)[k-1] > 0 {
			//           1 x 1 diagonal block
			//
			//           Invert the diagonal block.
			a.Set(k-1, k-1, cone/a.Get(k-1, k-1))

			//           Compute column K of the inverse.
			if k > 1 {
				goblas.Zcopy(toPtr(k-1), a.CVector(0, k-1), func() *int { y := 1; return &y }(), work, func() *int { y := 1; return &y }())
				Zsymv(uplo, toPtr(k-1), toPtrc128(-cone), a, lda, work, func() *int { y := 1; return &y }(), &czero, a.CVector(0, k-1), func() *int { y := 1; return &y }())
				a.Set(k-1, k-1, a.Get(k-1, k-1)-goblas.Zdotu(toPtr(k-1), work, func() *int { y := 1; return &y }(), a.CVector(0, k-1), func() *int { y := 1; return &y }()))
			}
			kstep = 1
		} else {
			//           2 x 2 diagonal block
			//
			//           Invert the diagonal block.
			t = a.Get(k-1, k+1-1)
			ak = a.Get(k-1, k-1) / t
			akp1 = a.Get(k+1-1, k+1-1) / t
			akkp1 = a.Get(k-1, k+1-1) / t
			d = t * (ak*akp1 - cone)
			a.Set(k-1, k-1, akp1/d)
			a.Set(k+1-1, k+1-1, ak/d)
			a.Set(k-1, k+1-1, -akkp1/d)

			//           Compute columns K and K+1 of the inverse.
			if k > 1 {
				goblas.Zcopy(toPtr(k-1), a.CVector(0, k-1), func() *int { y := 1; return &y }(), work, func() *int { y := 1; return &y }())
				Zsymv(uplo, toPtr(k-1), toPtrc128(-cone), a, lda, work, func() *int { y := 1; return &y }(), &czero, a.CVector(0, k-1), func() *int { y := 1; return &y }())
				a.Set(k-1, k-1, a.Get(k-1, k-1)-goblas.Zdotu(toPtr(k-1), work, func() *int { y := 1; return &y }(), a.CVector(0, k-1), func() *int { y := 1; return &y }()))
				a.Set(k-1, k+1-1, a.Get(k-1, k+1-1)-goblas.Zdotu(toPtr(k-1), a.CVector(0, k-1), func() *int { y := 1; return &y }(), a.CVector(0, k+1-1), func() *int { y := 1; return &y }()))
				goblas.Zcopy(toPtr(k-1), a.CVector(0, k+1-1), func() *int { y := 1; return &y }(), work, func() *int { y := 1; return &y }())
				Zsymv(uplo, toPtr(k-1), toPtrc128(-cone), a, lda, work, func() *int { y := 1; return &y }(), &czero, a.CVector(0, k+1-1), func() *int { y := 1; return &y }())
				a.Set(k+1-1, k+1-1, a.Get(k+1-1, k+1-1)-goblas.Zdotu(toPtr(k-1), work, func() *int { y := 1; return &y }(), a.CVector(0, k+1-1), func() *int { y := 1; return &y }()))
			}
			kstep = 2
		}

		if kstep == 1 {
			//           Interchange rows and columns K and IPIV(K) in the leading
			//           submatrix A(1:k+1,1:k+1)
			kp = (*ipiv)[k-1]
			if kp != k {
				if kp > 1 {
					goblas.Zswap(toPtr(kp-1), a.CVector(0, k-1), func() *int { y := 1; return &y }(), a.CVector(0, kp-1), func() *int { y := 1; return &y }())
				}
				goblas.Zswap(toPtr(k-kp-1), a.CVector(kp+1-1, k-1), func() *int { y := 1; return &y }(), a.CVector(kp-1, kp+1-1), lda)
				temp = a.Get(k-1, k-1)
				a.Set(k-1, k-1, a.Get(kp-1, kp-1))
				a.Set(kp-1, kp-1, temp)
			}
		} else {
			//           Interchange rows and columns K and K+1 with -IPIV(K) and
			//           -IPIV(K+1)in the leading submatrix A(1:k+1,1:k+1)
			kp = -(*ipiv)[k-1]
			if kp != k {
				if kp > 1 {
					goblas.Zswap(toPtr(kp-1), a.CVector(0, k-1), func() *int { y := 1; return &y }(), a.CVector(0, kp-1), func() *int { y := 1; return &y }())
				}
				goblas.Zswap(toPtr(k-kp-1), a.CVector(kp+1-1, k-1), func() *int { y := 1; return &y }(), a.CVector(kp-1, kp+1-1), lda)

				temp = a.Get(k-1, k-1)
				a.Set(k-1, k-1, a.Get(kp-1, kp-1))
				a.Set(kp-1, kp-1, temp)
				temp = a.Get(k-1, k+1-1)
				a.Set(k-1, k+1-1, a.Get(kp-1, k+1-1))
				a.Set(kp-1, k+1-1, temp)
			}

			k = k + 1
			kp = -(*ipiv)[k-1]
			if kp != k {
				if kp > 1 {
					goblas.Zswap(toPtr(kp-1), a.CVector(0, k-1), func() *int { y := 1; return &y }(), a.CVector(0, kp-1), func() *int { y := 1; return &y }())
				}
				goblas.Zswap(toPtr(k-kp-1), a.CVector(kp+1-1, k-1), func() *int { y := 1; return &y }(), a.CVector(kp-1, kp+1-1), lda)
				temp = a.Get(k-1, k-1)
				a.Set(k-1, k-1, a.Get(kp-1, kp-1))
				a.Set(kp-1, kp-1, temp)
			}
		}

		k = k + 1
		goto label30
	label40:
	} else {
		//        Compute inv(A) from the factorization A = L*D*L**T.
		//
		//        K is the main loop index, increasing from 1 to N in steps of
		//        1 or 2, depending on the size of the diagonal blocks.
		k = (*n)
	label50:
		;

		//        If K < 1, exit from loop.
		if k < 1 {
			goto label60
		}

		if (*ipiv)[k-1] > 0 {
			//           1 x 1 diagonal block
			//
			//           Invert the diagonal block.
			a.Set(k-1, k-1, cone/a.Get(k-1, k-1))

			//           Compute column K of the inverse.
			if k < (*n) {
				goblas.Zcopy(toPtr((*n)-k), a.CVector(k+1-1, k-1), func() *int { y := 1; return &y }(), work, func() *int { y := 1; return &y }())
				Zsymv(uplo, toPtr((*n)-k), toPtrc128(-cone), a.Off(k+1-1, k+1-1), lda, work, func() *int { y := 1; return &y }(), &czero, a.CVector(k+1-1, k-1), func() *int { y := 1; return &y }())
				a.Set(k-1, k-1, a.Get(k-1, k-1)-goblas.Zdotu(toPtr((*n)-k), work, func() *int { y := 1; return &y }(), a.CVector(k+1-1, k-1), func() *int { y := 1; return &y }()))
			}
			kstep = 1
		} else {
			//           2 x 2 diagonal block
			//
			//           Invert the diagonal block.
			t = a.Get(k-1, k-1-1)
			ak = a.Get(k-1-1, k-1-1) / t
			akp1 = a.Get(k-1, k-1) / t
			akkp1 = a.Get(k-1, k-1-1) / t
			d = t * (ak*akp1 - cone)
			a.Set(k-1-1, k-1-1, akp1/d)
			a.Set(k-1, k-1, ak/d)
			a.Set(k-1, k-1-1, -akkp1/d)

			//           Compute columns K-1 and K of the inverse.
			if k < (*n) {
				goblas.Zcopy(toPtr((*n)-k), a.CVector(k+1-1, k-1), func() *int { y := 1; return &y }(), work, func() *int { y := 1; return &y }())
				Zsymv(uplo, toPtr((*n)-k), toPtrc128(-cone), a.Off(k+1-1, k+1-1), lda, work, func() *int { y := 1; return &y }(), &czero, a.CVector(k+1-1, k-1), func() *int { y := 1; return &y }())
				a.Set(k-1, k-1, a.Get(k-1, k-1)-goblas.Zdotu(toPtr((*n)-k), work, func() *int { y := 1; return &y }(), a.CVector(k+1-1, k-1), func() *int { y := 1; return &y }()))
				a.Set(k-1, k-1-1, a.Get(k-1, k-1-1)-goblas.Zdotu(toPtr((*n)-k), a.CVector(k+1-1, k-1), func() *int { y := 1; return &y }(), a.CVector(k+1-1, k-1-1), func() *int { y := 1; return &y }()))
				goblas.Zcopy(toPtr((*n)-k), a.CVector(k+1-1, k-1-1), func() *int { y := 1; return &y }(), work, func() *int { y := 1; return &y }())
				Zsymv(uplo, toPtr((*n)-k), toPtrc128(-cone), a.Off(k+1-1, k+1-1), lda, work, func() *int { y := 1; return &y }(), &czero, a.CVector(k+1-1, k-1-1), func() *int { y := 1; return &y }())
				a.Set(k-1-1, k-1-1, a.Get(k-1-1, k-1-1)-goblas.Zdotu(toPtr((*n)-k), work, func() *int { y := 1; return &y }(), a.CVector(k+1-1, k-1-1), func() *int { y := 1; return &y }()))
			}
			kstep = 2
		}

		if kstep == 1 {
			//           Interchange rows and columns K and IPIV(K) in the trailing
			//           submatrix A(k-1:n,k-1:n)
			kp = (*ipiv)[k-1]
			if kp != k {
				if kp < (*n) {
					goblas.Zswap(toPtr((*n)-kp), a.CVector(kp+1-1, k-1), func() *int { y := 1; return &y }(), a.CVector(kp+1-1, kp-1), func() *int { y := 1; return &y }())
				}
				goblas.Zswap(toPtr(kp-k-1), a.CVector(k+1-1, k-1), func() *int { y := 1; return &y }(), a.CVector(kp-1, k+1-1), lda)
				temp = a.Get(k-1, k-1)
				a.Set(k-1, k-1, a.Get(kp-1, kp-1))
				a.Set(kp-1, kp-1, temp)
			}
		} else {
			//           Interchange rows and columns K and K-1 with -IPIV(K) and
			//           -IPIV(K-1) in the trailing submatrix A(k-1:n,k-1:n)
			kp = -(*ipiv)[k-1]
			if kp != k {
				if kp < (*n) {
					goblas.Zswap(toPtr((*n)-kp), a.CVector(kp+1-1, k-1), func() *int { y := 1; return &y }(), a.CVector(kp+1-1, kp-1), func() *int { y := 1; return &y }())
				}
				goblas.Zswap(toPtr(kp-k-1), a.CVector(k+1-1, k-1), func() *int { y := 1; return &y }(), a.CVector(kp-1, k+1-1), lda)

				temp = a.Get(k-1, k-1)
				a.Set(k-1, k-1, a.Get(kp-1, kp-1))
				a.Set(kp-1, kp-1, temp)
				temp = a.Get(k-1, k-1-1)
				a.Set(k-1, k-1-1, a.Get(kp-1, k-1-1))
				a.Set(kp-1, k-1-1, temp)
			}

			k = k - 1
			kp = -(*ipiv)[k-1]
			if kp != k {
				if kp < (*n) {
					goblas.Zswap(toPtr((*n)-kp), a.CVector(kp+1-1, k-1), func() *int { y := 1; return &y }(), a.CVector(kp+1-1, kp-1), func() *int { y := 1; return &y }())
				}
				goblas.Zswap(toPtr(kp-k-1), a.CVector(k+1-1, k-1), func() *int { y := 1; return &y }(), a.CVector(kp-1, k+1-1), lda)
				temp = a.Get(k-1, k-1)
				a.Set(k-1, k-1, a.Get(kp-1, kp-1))
				a.Set(kp-1, kp-1, temp)
			}
		}

		k = k - 1
		goto label50
	label60:
	}
}
