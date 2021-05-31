package golapack

import (
	"math/cmplx"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zhetrs solves a system of linear equations A*X = B with a complex
// Hermitian matrix A using the factorization A = U*D*U**H or
// A = L*D*L**H computed by ZHETRF.
func Zhetrs(uplo byte, n, nrhs *int, a *mat.CMatrix, lda *int, ipiv *[]int, b *mat.CMatrix, ldb, info *int) {
	var upper bool
	var ak, akm1, akm1k, bk, bkm1, denom, one complex128
	var s float64
	var j, k, kp int

	one = (1.0 + 0.0*1i)

	(*info) = 0
	upper = uplo == 'U'
	if !upper && uplo != 'L' {
		(*info) = -1
	} else if (*n) < 0 {
		(*info) = -2
	} else if (*nrhs) < 0 {
		(*info) = -3
	} else if (*lda) < maxint(1, *n) {
		(*info) = -5
	} else if (*ldb) < maxint(1, *n) {
		(*info) = -8
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("ZHETRS"), -(*info))
		return
	}

	//     Quick return if possible
	if (*n) == 0 || (*nrhs) == 0 {
		return
	}

	if upper {
		//        Solve A*X = B, where A = U*D*U**H.
		//
		//        First solve U*D*X = B, overwriting B with X.
		//
		//        K is the main loop index, decreasing from N to 1 in steps of
		//        1 or 2, depending on the size of the diagonal blocks.
		k = (*n)
	label10:
		;

		//        If K < 1, exit from loop.
		if k < 1 {
			goto label30
		}

		if (*ipiv)[k-1] > 0 {
			//           1 x 1 diagonal block
			//
			//           Interchange rows K and IPIV(K).
			kp = (*ipiv)[k-1]
			if kp != k {
				goblas.Zswap(nrhs, b.CVector(k-1, 0), ldb, b.CVector(kp-1, 0), ldb)
			}

			//           Multiply by inv(U(K)), where U(K) is the transformation
			//           stored in column K of A.
			goblas.Zgeru(toPtr(k-1), nrhs, toPtrc128(-one), a.CVector(0, k-1), func() *int { y := 1; return &y }(), b.CVector(k-1, 0), ldb, b, ldb)

			//           Multiply by the inverse of the diagonal block.
			s = real(one) / a.GetRe(k-1, k-1)
			goblas.Zdscal(nrhs, &s, b.CVector(k-1, 0), ldb)
			k = k - 1
		} else {
			//           2 x 2 diagonal block
			//
			//           Interchange rows K-1 and -IPIV(K).
			kp = -(*ipiv)[k-1]
			if kp != k-1 {
				goblas.Zswap(nrhs, b.CVector(k-1-1, 0), ldb, b.CVector(kp-1, 0), ldb)
			}

			//           Multiply by inv(U(K)), where U(K) is the transformation
			//           stored in columns K-1 and K of A.
			goblas.Zgeru(toPtr(k-2), nrhs, toPtrc128(-one), a.CVector(0, k-1), func() *int { y := 1; return &y }(), b.CVector(k-1, 0), ldb, b, ldb)
			goblas.Zgeru(toPtr(k-2), nrhs, toPtrc128(-one), a.CVector(0, k-1-1), func() *int { y := 1; return &y }(), b.CVector(k-1-1, 0), ldb, b, ldb)

			//           Multiply by the inverse of the diagonal block.
			akm1k = a.Get(k-1-1, k-1)
			akm1 = a.Get(k-1-1, k-1-1) / akm1k
			ak = a.Get(k-1, k-1) / cmplx.Conj(akm1k)
			denom = akm1*ak - one
			for j = 1; j <= (*nrhs); j++ {
				bkm1 = b.Get(k-1-1, j-1) / akm1k
				bk = b.Get(k-1, j-1) / cmplx.Conj(akm1k)
				b.Set(k-1-1, j-1, (ak*bkm1-bk)/denom)
				b.Set(k-1, j-1, (akm1*bk-bkm1)/denom)
			}
			k = k - 2
		}

		goto label10
	label30:
		;

		//        Next solve U**H *X = B, overwriting B with X.
		//
		//        K is the main loop index, increasing from 1 to N in steps of
		//        1 or 2, depending on the size of the diagonal blocks.
		k = 1
	label40:
		;

		//        If K > N, exit from loop.
		if k > (*n) {
			goto label50
		}

		if (*ipiv)[k-1] > 0 {
			//           1 x 1 diagonal block
			//
			//           Multiply by inv(U**H(K)), where U(K) is the transformation
			//           stored in column K of A.
			if k > 1 {
				Zlacgv(nrhs, b.CVector(k-1, 0), ldb)
				goblas.Zgemv(ConjTrans, toPtr(k-1), nrhs, toPtrc128(-one), b, ldb, a.CVector(0, k-1), func() *int { y := 1; return &y }(), &one, b.CVector(k-1, 0), ldb)
				Zlacgv(nrhs, b.CVector(k-1, 0), ldb)
			}

			//           Interchange rows K and IPIV(K).
			kp = (*ipiv)[k-1]
			if kp != k {
				goblas.Zswap(nrhs, b.CVector(k-1, 0), ldb, b.CVector(kp-1, 0), ldb)
			}
			k = k + 1
		} else {
			//           2 x 2 diagonal block
			//
			//           Multiply by inv(U**H(K+1)), where U(K+1) is the transformation
			//           stored in columns K and K+1 of A.
			if k > 1 {
				Zlacgv(nrhs, b.CVector(k-1, 0), ldb)
				goblas.Zgemv(ConjTrans, toPtr(k-1), nrhs, toPtrc128(-one), b, ldb, a.CVector(0, k-1), func() *int { y := 1; return &y }(), &one, b.CVector(k-1, 0), ldb)
				Zlacgv(nrhs, b.CVector(k-1, 0), ldb)
				//
				Zlacgv(nrhs, b.CVector(k+1-1, 0), ldb)
				goblas.Zgemv(ConjTrans, toPtr(k-1), nrhs, toPtrc128(-one), b, ldb, a.CVector(0, k+1-1), func() *int { y := 1; return &y }(), &one, b.CVector(k+1-1, 0), ldb)
				Zlacgv(nrhs, b.CVector(k+1-1, 0), ldb)
			}

			//           Interchange rows K and -IPIV(K).
			kp = -(*ipiv)[k-1]
			if kp != k {
				goblas.Zswap(nrhs, b.CVector(k-1, 0), ldb, b.CVector(kp-1, 0), ldb)
			}
			k = k + 2
		}

		goto label40
	label50:
	} else {
		//        Solve A*X = B, where A = L*D*L**H.
		//
		//        First solve L*D*X = B, overwriting B with X.
		//
		//        K is the main loop index, increasing from 1 to N in steps of
		//        1 or 2, depending on the size of the diagonal blocks.
		k = 1
	label60:
		;

		//        If K > N, exit from loop.
		if k > (*n) {
			goto label80
		}

		if (*ipiv)[k-1] > 0 {
			//           1 x 1 diagonal block
			//
			//           Interchange rows K and IPIV(K).
			kp = (*ipiv)[k-1]
			if kp != k {
				goblas.Zswap(nrhs, b.CVector(k-1, 0), ldb, b.CVector(kp-1, 0), ldb)
			}

			//           Multiply by inv(L(K)), where L(K) is the transformation
			//           stored in column K of A.
			if k < (*n) {
				goblas.Zgeru(toPtr((*n)-k), nrhs, toPtrc128(-one), a.CVector(k+1-1, k-1), func() *int { y := 1; return &y }(), b.CVector(k-1, 0), ldb, b.Off(k+1-1, 0), ldb)
			}

			//           Multiply by the inverse of the diagonal block.
			s = real(one) / a.GetRe(k-1, k-1)
			goblas.Zdscal(nrhs, &s, b.CVector(k-1, 0), ldb)
			k = k + 1
		} else {
			//           2 x 2 diagonal block
			//
			//           Interchange rows K+1 and -IPIV(K).
			kp = -(*ipiv)[k-1]
			if kp != k+1 {
				goblas.Zswap(nrhs, b.CVector(k+1-1, 0), ldb, b.CVector(kp-1, 0), ldb)
			}

			//           Multiply by inv(L(K)), where L(K) is the transformation
			//           stored in columns K and K+1 of A.
			if k < (*n)-1 {
				goblas.Zgeru(toPtr((*n)-k-1), nrhs, toPtrc128(-one), a.CVector(k+2-1, k-1), func() *int { y := 1; return &y }(), b.CVector(k-1, 0), ldb, b.Off(k+2-1, 0), ldb)
				goblas.Zgeru(toPtr((*n)-k-1), nrhs, toPtrc128(-one), a.CVector(k+2-1, k+1-1), func() *int { y := 1; return &y }(), b.CVector(k+1-1, 0), ldb, b.Off(k+2-1, 0), ldb)
			}

			//           Multiply by the inverse of the diagonal block.
			akm1k = a.Get(k+1-1, k-1)
			akm1 = a.Get(k-1, k-1) / cmplx.Conj(akm1k)
			ak = a.Get(k+1-1, k+1-1) / akm1k
			denom = akm1*ak - one
			for j = 1; j <= (*nrhs); j++ {
				bkm1 = b.Get(k-1, j-1) / cmplx.Conj(akm1k)
				bk = b.Get(k+1-1, j-1) / akm1k
				b.Set(k-1, j-1, (ak*bkm1-bk)/denom)
				b.Set(k+1-1, j-1, (akm1*bk-bkm1)/denom)
			}
			k = k + 2
		}

		goto label60
	label80:
		;

		//        Next solve L**H *X = B, overwriting B with X.
		//
		//        K is the main loop index, decreasing from N to 1 in steps of
		//        1 or 2, depending on the size of the diagonal blocks.
		k = (*n)
	label90:
		;

		//        If K < 1, exit from loop.
		if k < 1 {
			goto label100
		}

		if (*ipiv)[k-1] > 0 {
			//           1 x 1 diagonal block
			//
			//           Multiply by inv(L**H(K)), where L(K) is the transformation
			//           stored in column K of A.
			if k < (*n) {
				Zlacgv(nrhs, b.CVector(k-1, 0), ldb)
				goblas.Zgemv(ConjTrans, toPtr((*n)-k), nrhs, toPtrc128(-one), b.Off(k+1-1, 0), ldb, a.CVector(k+1-1, k-1), func() *int { y := 1; return &y }(), &one, b.CVector(k-1, 0), ldb)
				Zlacgv(nrhs, b.CVector(k-1, 0), ldb)
			}

			//           Interchange rows K and IPIV(K).
			kp = (*ipiv)[k-1]
			if kp != k {
				goblas.Zswap(nrhs, b.CVector(k-1, 0), ldb, b.CVector(kp-1, 0), ldb)
			}
			k = k - 1
		} else {
			//           2 x 2 diagonal block
			//
			//           Multiply by inv(L**H(K-1)), where L(K-1) is the transformation
			//           stored in columns K-1 and K of A.
			if k < (*n) {
				Zlacgv(nrhs, b.CVector(k-1, 0), ldb)
				goblas.Zgemv(ConjTrans, toPtr((*n)-k), nrhs, toPtrc128(-one), b.Off(k+1-1, 0), ldb, a.CVector(k+1-1, k-1), func() *int { y := 1; return &y }(), &one, b.CVector(k-1, 0), ldb)
				Zlacgv(nrhs, b.CVector(k-1, 0), ldb)
				//
				Zlacgv(nrhs, b.CVector(k-1-1, 0), ldb)
				goblas.Zgemv(ConjTrans, toPtr((*n)-k), nrhs, toPtrc128(-one), b.Off(k+1-1, 0), ldb, a.CVector(k+1-1, k-1-1), func() *int { y := 1; return &y }(), &one, b.CVector(k-1-1, 0), ldb)
				Zlacgv(nrhs, b.CVector(k-1-1, 0), ldb)
			}

			//           Interchange rows K and -IPIV(K).
			kp = -(*ipiv)[k-1]
			if kp != k {
				goblas.Zswap(nrhs, b.CVector(k-1, 0), ldb, b.CVector(kp-1, 0), ldb)
			}
			k = k - 2
		}

		goto label90
	label100:
	}
}
