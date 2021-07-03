package golapack

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zsytrsrook solves a system of linear equations A*X = B with
// a complex symmetric matrix A using the factorization A = U*D*U**T or
// A = L*D*L**T computed by ZSYTRF_ROOK.
func Zsytrsrook(uplo byte, n, nrhs *int, a *mat.CMatrix, lda *int, ipiv *[]int, b *mat.CMatrix, ldb, info *int) {
	var upper bool
	var ak, akm1, akm1k, bk, bkm1, cone, denom complex128
	var j, k, kp int
	var err error
	_ = err

	cone = (1.0 + 0.0*1i)

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
		gltest.Xerbla([]byte("ZSYTRS_ROOK"), -(*info))
		return
	}

	//     Quick return if possible
	if (*n) == 0 || (*nrhs) == 0 {
		return
	}

	if upper {
		//        Solve A*X = B, where A = U*D*U**T.
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
				goblas.Zswap(*nrhs, b.CVector(k-1, 0), *ldb, b.CVector(kp-1, 0), *ldb)
			}

			//           Multiply by inv(U(K)), where U(K) is the transformation
			//           stored in column K of A.
			err = goblas.Zgeru(k-1, *nrhs, -cone, a.CVector(0, k-1), 1, b.CVector(k-1, 0), *ldb, b, *ldb)

			//           Multiply by the inverse of the diagonal block.
			goblas.Zscal(*nrhs, cone/a.Get(k-1, k-1), b.CVector(k-1, 0), *ldb)
			k = k - 1
		} else {
			//           2 x 2 diagonal block
			//
			//           Interchange rows K and -IPIV(K) THEN K-1 and -IPIV(K-1)
			kp = -(*ipiv)[k-1]
			if kp != k {
				goblas.Zswap(*nrhs, b.CVector(k-1, 0), *ldb, b.CVector(kp-1, 0), *ldb)
			}

			kp = -(*ipiv)[k-1-1]
			if kp != k-1 {
				goblas.Zswap(*nrhs, b.CVector(k-1-1, 0), *ldb, b.CVector(kp-1, 0), *ldb)
			}

			//           Multiply by inv(U(K)), where U(K) is the transformation
			//           stored in columns K-1 and K of A.
			if k > 2 {
				err = goblas.Zgeru(k-2, *nrhs, -cone, a.CVector(0, k-1), 1, b.CVector(k-1, 0), *ldb, b, *ldb)
				err = goblas.Zgeru(k-2, *nrhs, -cone, a.CVector(0, k-1-1), 1, b.CVector(k-1-1, 0), *ldb, b, *ldb)
			}

			//           Multiply by the inverse of the diagonal block.
			akm1k = a.Get(k-1-1, k-1)
			akm1 = a.Get(k-1-1, k-1-1) / akm1k
			ak = a.Get(k-1, k-1) / akm1k
			denom = akm1*ak - cone
			for j = 1; j <= (*nrhs); j++ {
				bkm1 = b.Get(k-1-1, j-1) / akm1k
				bk = b.Get(k-1, j-1) / akm1k
				b.Set(k-1-1, j-1, (ak*bkm1-bk)/denom)
				b.Set(k-1, j-1, (akm1*bk-bkm1)/denom)
			}
			k = k - 2
		}

		goto label10
	label30:
		;

		//        Next solve U**T *X = B, overwriting B with X.
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
			//           Multiply by inv(U**T(K)), where U(K) is the transformation
			//           stored in column K of A.
			if k > 1 {
				err = goblas.Zgemv(Trans, k-1, *nrhs, -cone, b, *ldb, a.CVector(0, k-1), 1, cone, b.CVector(k-1, 0), *ldb)
			}

			//           Interchange rows K and IPIV(K).
			kp = (*ipiv)[k-1]
			if kp != k {
				goblas.Zswap(*nrhs, b.CVector(k-1, 0), *ldb, b.CVector(kp-1, 0), *ldb)
			}
			k = k + 1
		} else {
			//           2 x 2 diagonal block
			//
			//           Multiply by inv(U**T(K+1)), where U(K+1) is the transformation
			//           stored in columns K and K+1 of A.
			if k > 1 {
				err = goblas.Zgemv(Trans, k-1, *nrhs, -cone, b, *ldb, a.CVector(0, k-1), 1, cone, b.CVector(k-1, 0), *ldb)
				err = goblas.Zgemv(Trans, k-1, *nrhs, -cone, b, *ldb, a.CVector(0, k+1-1), 1, cone, b.CVector(k+1-1, 0), *ldb)
			}

			//           Interchange rows K and -IPIV(K) THEN K+1 and -IPIV(K+1).
			kp = -(*ipiv)[k-1]
			if kp != k {
				goblas.Zswap(*nrhs, b.CVector(k-1, 0), *ldb, b.CVector(kp-1, 0), *ldb)
			}

			kp = -(*ipiv)[k+1-1]
			if kp != k+1 {
				goblas.Zswap(*nrhs, b.CVector(k+1-1, 0), *ldb, b.CVector(kp-1, 0), *ldb)
			}

			k = k + 2
		}

		goto label40
	label50:
	} else {
		//        Solve A*X = B, where A = L*D*L**T.
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
				goblas.Zswap(*nrhs, b.CVector(k-1, 0), *ldb, b.CVector(kp-1, 0), *ldb)
			}

			//           Multiply by inv(L(K)), where L(K) is the transformation
			//           stored in column K of A.
			if k < (*n) {
				err = goblas.Zgeru((*n)-k, *nrhs, -cone, a.CVector(k+1-1, k-1), 1, b.CVector(k-1, 0), *ldb, b.Off(k+1-1, 0), *ldb)
			}

			//           Multiply by the inverse of the diagonal block.
			goblas.Zscal(*nrhs, cone/a.Get(k-1, k-1), b.CVector(k-1, 0), *ldb)
			k = k + 1
		} else {
			//           2 x 2 diagonal block
			//
			//           Interchange rows K and -IPIV(K) THEN K+1 and -IPIV(K+1)
			kp = -(*ipiv)[k-1]
			if kp != k {
				goblas.Zswap(*nrhs, b.CVector(k-1, 0), *ldb, b.CVector(kp-1, 0), *ldb)
			}

			kp = -(*ipiv)[k+1-1]
			if kp != k+1 {
				goblas.Zswap(*nrhs, b.CVector(k+1-1, 0), *ldb, b.CVector(kp-1, 0), *ldb)
			}

			//           Multiply by inv(L(K)), where L(K) is the transformation
			//           stored in columns K and K+1 of A.
			if k < (*n)-1 {
				err = goblas.Zgeru((*n)-k-1, *nrhs, -cone, a.CVector(k+2-1, k-1), 1, b.CVector(k-1, 0), *ldb, b.Off(k+2-1, 0), *ldb)
				err = goblas.Zgeru((*n)-k-1, *nrhs, -cone, a.CVector(k+2-1, k+1-1), 1, b.CVector(k+1-1, 0), *ldb, b.Off(k+2-1, 0), *ldb)
			}

			//           Multiply by the inverse of the diagonal block.
			akm1k = a.Get(k+1-1, k-1)
			akm1 = a.Get(k-1, k-1) / akm1k
			ak = a.Get(k+1-1, k+1-1) / akm1k
			denom = akm1*ak - cone
			for j = 1; j <= (*nrhs); j++ {
				bkm1 = b.Get(k-1, j-1) / akm1k
				bk = b.Get(k+1-1, j-1) / akm1k
				b.Set(k-1, j-1, (ak*bkm1-bk)/denom)
				b.Set(k+1-1, j-1, (akm1*bk-bkm1)/denom)
			}
			k = k + 2
		}

		goto label60
	label80:
		;

		//        Next solve L**T *X = B, overwriting B with X.
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
			//           Multiply by inv(L**T(K)), where L(K) is the transformation
			//           stored in column K of A.
			if k < (*n) {
				err = goblas.Zgemv(Trans, (*n)-k, *nrhs, -cone, b.Off(k+1-1, 0), *ldb, a.CVector(k+1-1, k-1), 1, cone, b.CVector(k-1, 0), *ldb)
			}

			//           Interchange rows K and IPIV(K).
			kp = (*ipiv)[k-1]
			if kp != k {
				goblas.Zswap(*nrhs, b.CVector(k-1, 0), *ldb, b.CVector(kp-1, 0), *ldb)
			}
			k = k - 1
		} else {
			//           2 x 2 diagonal block
			//
			//           Multiply by inv(L**T(K-1)), where L(K-1) is the transformation
			//           stored in columns K-1 and K of A.
			if k < (*n) {
				err = goblas.Zgemv(Trans, (*n)-k, *nrhs, -cone, b.Off(k+1-1, 0), *ldb, a.CVector(k+1-1, k-1), 1, cone, b.CVector(k-1, 0), *ldb)
				err = goblas.Zgemv(Trans, (*n)-k, *nrhs, -cone, b.Off(k+1-1, 0), *ldb, a.CVector(k+1-1, k-1-1), 1, cone, b.CVector(k-1-1, 0), *ldb)
			}

			//           Interchange rows K and -IPIV(K) THEN K-1 and -IPIV(K-1)
			kp = -(*ipiv)[k-1]
			if kp != k {
				goblas.Zswap(*nrhs, b.CVector(k-1, 0), *ldb, b.CVector(kp-1, 0), *ldb)
			}

			kp = -(*ipiv)[k-1-1]
			if kp != k-1 {
				goblas.Zswap(*nrhs, b.CVector(k-1-1, 0), *ldb, b.CVector(kp-1, 0), *ldb)
			}

			k = k - 2
		}

		goto label90
	label100:
	}
}
