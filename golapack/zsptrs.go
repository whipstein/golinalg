package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zsptrs solves a system of linear equations A*X = B with a complex
// symmetric matrix A stored in packed format using the factorization
// A = U*D*U**T or A = L*D*L**T computed by ZSPTRF.
func Zsptrs(uplo mat.MatUplo, n, nrhs int, ap *mat.CVector, ipiv *[]int, b *mat.CMatrix) (err error) {
	var upper bool
	var ak, akm1, akm1k, bk, bkm1, denom, one complex128
	var j, k, kc, kp int

	one = (1.0 + 0.0*1i)

	upper = uplo == Upper
	if !upper && uplo != Lower {
		err = fmt.Errorf("!upper && uplo != Lower: uplo=%s", uplo)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if nrhs < 0 {
		err = fmt.Errorf("nrhs < 0: nrhs=%v", nrhs)
	} else if b.Rows < max(1, n) {
		err = fmt.Errorf("b.Rows < max(1, n): b.Row=%v, n=%v", b.Rows, n)
	}
	if err != nil {
		gltest.Xerbla2("Zsptrs", err)
		return
	}

	//     Quick return if possible
	if n == 0 || nrhs == 0 {
		return
	}

	if upper {
		//        Solve A*X = B, where A = U*D*U**T.
		//
		//        First solve U*D*X = B, overwriting B with X.
		//
		//        K is the main loop index, decreasing from N to 1 in steps of
		//        1 or 2, depending on the size of the diagonal blocks.
		k = n
		kc = n*(n+1)/2 + 1
	label10:
		;

		//        If K < 1, exit from loop.
		if k < 1 {
			goto label30
		}

		kc = kc - k
		if (*ipiv)[k-1] > 0 {
			//           1 x 1 diagonal block
			//
			//           Interchange rows K and IPIV(K).
			kp = (*ipiv)[k-1]
			if kp != k {
				b.Off(kp-1, 0).CVector().Swap(nrhs, b.Off(k-1, 0).CVector(), b.Rows, b.Rows)
			}

			//           Multiply by inv(U(K)), where U(K) is the transformation
			//           stored in column K of A.
			err = b.Geru(k-1, nrhs, -one, ap.Off(kc-1), 1, b.Off(k-1, 0).CVector(), b.Rows)

			//           Multiply by the inverse of the diagonal block.
			b.Off(k-1, 0).CVector().Scal(nrhs, one/ap.Get(kc+k-1-1), b.Rows)
			k = k - 1
		} else {
			//           2 x 2 diagonal block
			//
			//           Interchange rows K-1 and -IPIV(K).
			kp = -(*ipiv)[k-1]
			if kp != k-1 {
				b.Off(kp-1, 0).CVector().Swap(nrhs, b.Off(k-1-1, 0).CVector(), b.Rows, b.Rows)
			}

			//           Multiply by inv(U(K)), where U(K) is the transformation
			//           stored in columns K-1 and K of A.
			err = b.Geru(k-2, nrhs, -one, ap.Off(kc-1), 1, b.Off(k-1, 0).CVector(), b.Rows)
			err = b.Geru(k-2, nrhs, -one, ap.Off(kc-(k-1)-1), 1, b.Off(k-1-1, 0).CVector(), b.Rows)

			//           Multiply by the inverse of the diagonal block.
			akm1k = ap.Get(kc + k - 2 - 1)
			akm1 = ap.Get(kc-1-1) / akm1k
			ak = ap.Get(kc+k-1-1) / akm1k
			denom = akm1*ak - one
			for j = 1; j <= nrhs; j++ {
				bkm1 = b.Get(k-1-1, j-1) / akm1k
				bk = b.Get(k-1, j-1) / akm1k
				b.Set(k-1-1, j-1, (ak*bkm1-bk)/denom)
				b.Set(k-1, j-1, (akm1*bk-bkm1)/denom)
			}
			kc = kc - k + 1
			k = k - 2
		}

		goto label10
	label30:
		;

		//        Next solve U**T*X = B, overwriting B with X.
		//
		//        K is the main loop index, increasing from 1 to N in steps of
		//        1 or 2, depending on the size of the diagonal blocks.
		k = 1
		kc = 1
	label40:
		;

		//        If K > N, exit from loop.
		if k > n {
			goto label50
		}

		if (*ipiv)[k-1] > 0 {
			//           1 x 1 diagonal block
			//
			//           Multiply by inv(U**T(K)), where U(K) is the transformation
			//           stored in column K of A.
			err = b.Off(k-1, 0).CVector().Gemv(Trans, k-1, nrhs, -one, b, ap.Off(kc-1), 1, one, b.Rows)

			//           Interchange rows K and IPIV(K).
			kp = (*ipiv)[k-1]
			if kp != k {
				b.Off(kp-1, 0).CVector().Swap(nrhs, b.Off(k-1, 0).CVector(), b.Rows, b.Rows)
			}
			kc = kc + k
			k = k + 1
		} else {
			//           2 x 2 diagonal block
			//
			//           Multiply by inv(U**T(K+1)), where U(K+1) is the transformation
			//           stored in columns K and K+1 of A.
			err = b.Off(k-1, 0).CVector().Gemv(Trans, k-1, nrhs, -one, b, ap.Off(kc-1), 1, one, b.Rows)
			err = b.Off(k, 0).CVector().Gemv(Trans, k-1, nrhs, -one, b, ap.Off(kc+k-1), 1, one, b.Rows)

			//           Interchange rows K and -IPIV(K).
			kp = -(*ipiv)[k-1]
			if kp != k {
				b.Off(kp-1, 0).CVector().Swap(nrhs, b.Off(k-1, 0).CVector(), b.Rows, b.Rows)
			}
			kc = kc + 2*k + 1
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
		kc = 1
	label60:
		;

		//        If K > N, exit from loop.
		if k > n {
			goto label80
		}

		if (*ipiv)[k-1] > 0 {
			//           1 x 1 diagonal block
			//
			//           Interchange rows K and IPIV(K).
			kp = (*ipiv)[k-1]
			if kp != k {
				b.Off(kp-1, 0).CVector().Swap(nrhs, b.Off(k-1, 0).CVector(), b.Rows, b.Rows)
			}

			//           Multiply by inv(L(K)), where L(K) is the transformation
			//           stored in column K of A.
			if k < n {
				err = b.Off(k, 0).Geru(n-k, nrhs, -one, ap.Off(kc), 1, b.Off(k-1, 0).CVector(), b.Rows)
			}

			//           Multiply by the inverse of the diagonal block.
			b.Off(k-1, 0).CVector().Scal(nrhs, one/ap.Get(kc-1), b.Rows)
			kc = kc + n - k + 1
			k = k + 1
		} else {
			//           2 x 2 diagonal block
			//
			//           Interchange rows K+1 and -IPIV(K).
			kp = -(*ipiv)[k-1]
			if kp != k+1 {
				b.Off(kp-1, 0).CVector().Swap(nrhs, b.Off(k, 0).CVector(), b.Rows, b.Rows)
			}

			//           Multiply by inv(L(K)), where L(K) is the transformation
			//           stored in columns K and K+1 of A.
			if k < n-1 {
				err = b.Off(k+2-1, 0).Geru(n-k-1, nrhs, -one, ap.Off(kc+2-1), 1, b.Off(k-1, 0).CVector(), b.Rows)
				err = b.Off(k+2-1, 0).Geru(n-k-1, nrhs, -one, ap.Off(kc+n-k+2-1), 1, b.Off(k, 0).CVector(), b.Rows)
			}

			//           Multiply by the inverse of the diagonal block.
			akm1k = ap.Get(kc + 1 - 1)
			akm1 = ap.Get(kc-1) / akm1k
			ak = ap.Get(kc+n-k) / akm1k
			denom = akm1*ak - one
			for j = 1; j <= nrhs; j++ {
				bkm1 = b.Get(k-1, j-1) / akm1k
				bk = b.Get(k, j-1) / akm1k
				b.Set(k-1, j-1, (ak*bkm1-bk)/denom)
				b.Set(k, j-1, (akm1*bk-bkm1)/denom)
			}
			kc = kc + 2*(n-k) + 1
			k = k + 2
		}

		goto label60
	label80:
		;

		//        Next solve L**T*X = B, overwriting B with X.
		//
		//        K is the main loop index, decreasing from N to 1 in steps of
		//        1 or 2, depending on the size of the diagonal blocks.
		k = n
		kc = n*(n+1)/2 + 1
	label90:
		;

		//        If K < 1, exit from loop.
		if k < 1 {
			goto label100
		}

		kc = kc - (n - k + 1)
		if (*ipiv)[k-1] > 0 {
			//           1 x 1 diagonal block
			//
			//           Multiply by inv(L**T(K)), where L(K) is the transformation
			//           stored in column K of A.
			if k < n {
				err = b.Off(k-1, 0).CVector().Gemv(Trans, n-k, nrhs, -one, b.Off(k, 0), ap.Off(kc), 1, one, b.Rows)
			}

			//           Interchange rows K and IPIV(K).
			kp = (*ipiv)[k-1]
			if kp != k {
				b.Off(kp-1, 0).CVector().Swap(nrhs, b.Off(k-1, 0).CVector(), b.Rows, b.Rows)
			}
			k = k - 1
		} else {
			//           2 x 2 diagonal block
			//
			//           Multiply by inv(L**T(K-1)), where L(K-1) is the transformation
			//           stored in columns K-1 and K of A.
			if k < n {
				err = b.Off(k-1, 0).CVector().Gemv(Trans, n-k, nrhs, -one, b.Off(k, 0), ap.Off(kc), 1, one, b.Rows)
				err = b.Off(k-1-1, 0).CVector().Gemv(Trans, n-k, nrhs, -one, b.Off(k, 0), ap.Off(kc-(n-k)-1), 1, one, b.Rows)
			}

			//           Interchange rows K and -IPIV(K).
			kp = -(*ipiv)[k-1]
			if kp != k {
				b.Off(kp-1, 0).CVector().Swap(nrhs, b.Off(k-1, 0).CVector(), b.Rows, b.Rows)
			}
			kc = kc - (n - k + 2)
			k = k - 2
		}

		goto label90
	label100:
	}

	return
}
