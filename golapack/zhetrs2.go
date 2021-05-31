package golapack

import (
	"math/cmplx"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zhetrs2 solves a system of linear equations A*X = B with a complex
// Hermitian matrix A using the factorization A = U*D*U**H or
// A = L*D*L**H computed by ZHETRF and converted by ZSYCONV.
func Zhetrs2(uplo byte, n, nrhs *int, a *mat.CMatrix, lda *int, ipiv *[]int, b *mat.CMatrix, ldb *int, work *mat.CVector, info *int) {
	var upper bool
	var ak, akm1, akm1k, bk, bkm1, denom, one complex128
	var s float64
	var i, iinfo, j, k, kp int

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
		gltest.Xerbla([]byte("ZHETRS2"), -(*info))
		return
	}

	//     Quick return if possible
	if (*n) == 0 || (*nrhs) == 0 {
		return
	}

	//     Convert A
	Zsyconv(uplo, 'C', n, a, lda, ipiv, work, &iinfo)

	if upper {
		//        Solve A*X = B, where A = U*D*U**H.
		//
		//       P**T * B
		k = (*n)
		for k >= 1 {
			if (*ipiv)[k-1] > 0 {
				//           1 x 1 diagonal block
				//           Interchange rows K and IPIV(K).
				kp = (*ipiv)[k-1]
				if kp != k {
					goblas.Zswap(nrhs, b.CVector(k-1, 0), ldb, b.CVector(kp-1, 0), ldb)
				}
				k = k - 1
			} else {
				//           2 x 2 diagonal block
				//           Interchange rows K-1 and -IPIV(K).
				kp = -(*ipiv)[k-1]
				if kp == -(*ipiv)[k-1-1] {
					goblas.Zswap(nrhs, b.CVector(k-1-1, 0), ldb, b.CVector(kp-1, 0), ldb)
				}
				k = k - 2
			}
		}

		//  Compute (U \P**T * B) -> B    [ (U \P**T * B) ]
		goblas.Ztrsm(Left, Upper, NoTrans, Unit, n, nrhs, &one, a, lda, b, ldb)

		//  Compute D \ B -> B   [ D \ (U \P**T * B) ]
		i = (*n)
		for i >= 1 {
			if (*ipiv)[i-1] > 0 {
				s = real(one) / a.GetRe(i-1, i-1)
				goblas.Zdscal(nrhs, &s, b.CVector(i-1, 0), ldb)
			} else if i > 1 {
				if (*ipiv)[i-1-1] == (*ipiv)[i-1] {
					akm1k = work.Get(i - 1)
					akm1 = a.Get(i-1-1, i-1-1) / akm1k
					ak = a.Get(i-1, i-1) / cmplx.Conj(akm1k)
					denom = akm1*ak - one
					for j = 1; j <= (*nrhs); j++ {
						bkm1 = b.Get(i-1-1, j-1) / akm1k
						bk = b.Get(i-1, j-1) / cmplx.Conj(akm1k)
						b.Set(i-1-1, j-1, (ak*bkm1-bk)/denom)
						b.Set(i-1, j-1, (akm1*bk-bkm1)/denom)
					}
					i = i - 1
				}
			}
			i = i - 1
		}

		//      Compute (U**H \ B) -> B   [ U**H \ (D \ (U \P**T * B) ) ]
		goblas.Ztrsm(Left, Upper, ConjTrans, Unit, n, nrhs, &one, a, lda, b, ldb)

		//       P * B  [ P * (U**H \ (D \ (U \P**T * B) )) ]
		k = 1
		for k <= (*n) {
			if (*ipiv)[k-1] > 0 {
				//           1 x 1 diagonal block
				//           Interchange rows K and IPIV(K).
				kp = (*ipiv)[k-1]
				if kp != k {
					goblas.Zswap(nrhs, b.CVector(k-1, 0), ldb, b.CVector(kp-1, 0), ldb)
				}
				k = k + 1
			} else {
				//           2 x 2 diagonal block
				//           Interchange rows K-1 and -IPIV(K).
				kp = -(*ipiv)[k-1]
				if k < (*n) && kp == -(*ipiv)[k+1-1] {
					goblas.Zswap(nrhs, b.CVector(k-1, 0), ldb, b.CVector(kp-1, 0), ldb)
				}
				k = k + 2
			}
		}

	} else {
		//        Solve A*X = B, where A = L*D*L**H.
		//
		//       P**T * B
		k = 1
		for k <= (*n) {
			if (*ipiv)[k-1] > 0 {
				//           1 x 1 diagonal block
				//           Interchange rows K and IPIV(K).
				kp = (*ipiv)[k-1]
				if kp != k {
					goblas.Zswap(nrhs, b.CVector(k-1, 0), ldb, b.CVector(kp-1, 0), ldb)
				}
				k = k + 1
			} else {
				//           2 x 2 diagonal block
				//           Interchange rows K and -IPIV(K+1).
				kp = -(*ipiv)[k+1-1]
				if kp == -(*ipiv)[k-1] {
					goblas.Zswap(nrhs, b.CVector(k+1-1, 0), ldb, b.CVector(kp-1, 0), ldb)
				}
				k = k + 2
			}
		}

		//  Compute (L \P**T * B) -> B    [ (L \P**T * B) ]
		goblas.Ztrsm(Left, Lower, NoTrans, Unit, n, nrhs, &one, a, lda, b, ldb)

		//  Compute D \ B -> B   [ D \ (L \P**T * B) ]
		i = 1
		for i <= (*n) {
			if (*ipiv)[i-1] > 0 {
				s = real(one) / a.GetRe(i-1, i-1)
				goblas.Zdscal(nrhs, &s, b.CVector(i-1, 0), ldb)
			} else {
				akm1k = work.Get(i - 1)
				akm1 = a.Get(i-1, i-1) / cmplx.Conj(akm1k)
				ak = a.Get(i+1-1, i+1-1) / akm1k
				denom = akm1*ak - one
				for j = 1; j <= (*nrhs); j++ {
					bkm1 = b.Get(i-1, j-1) / cmplx.Conj(akm1k)
					bk = b.Get(i+1-1, j-1) / akm1k
					b.Set(i-1, j-1, (ak*bkm1-bk)/denom)
					b.Set(i+1-1, j-1, (akm1*bk-bkm1)/denom)
				}
				i = i + 1
			}
			i = i + 1
		}

		//  Compute (L**H \ B) -> B   [ L**H \ (D \ (L \P**T * B) ) ]
		goblas.Ztrsm(Left, Lower, ConjTrans, Unit, n, nrhs, &one, a, lda, b, ldb)

		//       P * B  [ P * (L**H \ (D \ (L \P**T * B) )) ]
		k = (*n)
		for k >= 1 {
			if (*ipiv)[k-1] > 0 {
				//           1 x 1 diagonal block
				//           Interchange rows K and IPIV(K).
				kp = (*ipiv)[k-1]
				if kp != k {
					goblas.Zswap(nrhs, b.CVector(k-1, 0), ldb, b.CVector(kp-1, 0), ldb)
				}
				k = k - 1
			} else {
				//           2 x 2 diagonal block
				//           Interchange rows K-1 and -IPIV(K).
				kp = -(*ipiv)[k-1]
				if k > 1 && kp == -(*ipiv)[k-1-1] {
					goblas.Zswap(nrhs, b.CVector(k-1, 0), ldb, b.CVector(kp-1, 0), ldb)
				}
				k = k - 2
			}
		}

	}

	//     Revert A
	Zsyconv(uplo, 'R', n, a, lda, ipiv, work, &iinfo)
}
