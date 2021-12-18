package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// DsytrsRook solves a system of linear equations A*X = B with
// a real symmetric matrix A using the factorization A = U*D*U**T or
// A = L*D*L**T computed by DSYTRF_ROOK.
func DsytrsRook(uplo mat.MatUplo, n, nrhs int, a *mat.Matrix, ipiv *[]int, b *mat.Matrix) (err error) {
	var upper bool
	var ak, akm1, akm1k, bk, bkm1, denom, one float64
	var j, k, kp int

	one = 1.0

	upper = uplo == Upper
	if !upper && uplo != Lower {
		err = fmt.Errorf("!upper && uplo != Lower: uplo=%s", uplo)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if nrhs < 0 {
		err = fmt.Errorf("nrhs < 0: nrhs=%v", nrhs)
	} else if a.Rows < max(1, n) {
		err = fmt.Errorf("a.Rows < max(1, n): a.Rows=%v, n=%v", a.Rows, n)
	} else if b.Rows < max(1, n) {
		err = fmt.Errorf("b.Rows < max(1, n): b.Rows=%v, n=%v", b.Rows, n)
	}
	if err != nil {
		gltest.Xerbla2("DsytrsRook", err)
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
				b.Off(kp-1, 0).Vector().Swap(nrhs, b.Off(k-1, 0).Vector(), b.Rows, b.Rows)
			}

			//           Multiply by inv(U(K)), where U(K) is the transformation
			//           stored in column K of A.
			if err = b.Ger(k-1, nrhs, -one, a.Off(0, k-1).Vector(), 1, b.Off(k-1, 0).Vector(), b.Rows); err != nil {
				panic(err)
			}

			//           Multiply by the inverse of the diagonal block.
			b.Off(k-1, 0).Vector().Scal(nrhs, one/a.Get(k-1, k-1), b.Rows)
			k = k - 1
		} else {
			//           2 x 2 diagonal block
			//
			//           Interchange rows K and -IPIV(K) THEN K-1 and -IPIV(K-1)
			kp = -(*ipiv)[k-1]
			if kp != k {
				b.Off(kp-1, 0).Vector().Swap(nrhs, b.Off(k-1, 0).Vector(), b.Rows, b.Rows)
			}

			kp = -(*ipiv)[k-1-1]
			if kp != k-1 {
				b.Off(kp-1, 0).Vector().Swap(nrhs, b.Off(k-1-1, 0).Vector(), b.Rows, b.Rows)
			}

			//           Multiply by inv(U(K)), where U(K) is the transformation
			//           stored in columns K-1 and K of A.
			if k > 2 {
				if err = b.Ger(k-2, nrhs, -one, a.Off(0, k-1).Vector(), 1, b.Off(k-1, 0).Vector(), b.Rows); err != nil {
					panic(err)
				}
				if err = b.Ger(k-2, nrhs, -one, a.Off(0, k-1-1).Vector(), 1, b.Off(k-1-1, 0).Vector(), b.Rows); err != nil {
					panic(err)
				}
			}

			//           Multiply by the inverse of the diagonal block.
			akm1k = a.Get(k-1-1, k-1)
			akm1 = a.Get(k-1-1, k-1-1) / akm1k
			ak = a.Get(k-1, k-1) / akm1k
			denom = akm1*ak - one
			for j = 1; j <= nrhs; j++ {
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
		if k > n {
			goto label50
		}

		if (*ipiv)[k-1] > 0 {
			//           1 x 1 diagonal block
			//
			//           Multiply by inv(U**T(K)), where U(K) is the transformation
			//           stored in column K of A.
			if k > 1 {
				if err = b.Off(k-1, 0).Vector().Gemv(Trans, k-1, nrhs, -one, b, a.Off(0, k-1).Vector(), 1, one, b.Rows); err != nil {
					panic(err)
				}
			}

			//           Interchange rows K and IPIV(K).
			kp = (*ipiv)[k-1]
			if kp != k {
				b.Off(kp-1, 0).Vector().Swap(nrhs, b.Off(k-1, 0).Vector(), b.Rows, b.Rows)
			}
			k = k + 1
		} else {
			//           2 x 2 diagonal block
			//
			//           Multiply by inv(U**T(K+1)), where U(K+1) is the transformation
			//           stored in columns K and K+1 of A.
			if k > 1 {
				if err = b.Off(k-1, 0).Vector().Gemv(Trans, k-1, nrhs, -one, b, a.Off(0, k-1).Vector(), 1, one, b.Rows); err != nil {
					panic(err)
				}
				if err = b.Off(k, 0).Vector().Gemv(Trans, k-1, nrhs, -one, b, a.Off(0, k).Vector(), 1, one, b.Rows); err != nil {
					panic(err)
				}
			}

			//           Interchange rows K and -IPIV(K) THEN K+1 and -IPIV(K+1).
			kp = -(*ipiv)[k-1]
			if kp != k {
				b.Off(kp-1, 0).Vector().Swap(nrhs, b.Off(k-1, 0).Vector(), b.Rows, b.Rows)
			}

			kp = -(*ipiv)[k]
			if kp != k+1 {
				b.Off(kp-1, 0).Vector().Swap(nrhs, b.Off(k, 0).Vector(), b.Rows, b.Rows)
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
		if k > n {
			goto label80
		}

		if (*ipiv)[k-1] > 0 {
			//           1 x 1 diagonal block
			//
			//           Interchange rows K and IPIV(K).
			kp = (*ipiv)[k-1]
			if kp != k {
				b.Off(kp-1, 0).Vector().Swap(nrhs, b.Off(k-1, 0).Vector(), b.Rows, b.Rows)
			}

			//           Multiply by inv(L(K)), where L(K) is the transformation
			//           stored in column K of A.
			if k < n {
				if err = b.Off(k, 0).Ger(n-k, nrhs, -one, a.Off(k, k-1).Vector(), 1, b.Off(k-1, 0).Vector(), b.Rows); err != nil {
					panic(err)
				}
			}

			//           Multiply by the inverse of the diagonal block.
			b.Off(k-1, 0).Vector().Scal(nrhs, one/a.Get(k-1, k-1), b.Rows)
			k = k + 1
		} else {
			//           2 x 2 diagonal block
			//
			//           Interchange rows K and -IPIV(K) THEN K+1 and -IPIV(K+1)
			kp = -(*ipiv)[k-1]
			if kp != k {
				b.Off(kp-1, 0).Vector().Swap(nrhs, b.Off(k-1, 0).Vector(), b.Rows, b.Rows)
			}

			kp = -(*ipiv)[k]
			if kp != k+1 {
				b.Off(kp-1, 0).Vector().Swap(nrhs, b.Off(k, 0).Vector(), b.Rows, b.Rows)
			}

			//           Multiply by inv(L(K)), where L(K) is the transformation
			//           stored in columns K and K+1 of A.
			if k < n-1 {
				if err = b.Off(k+2-1, 0).Ger(n-k-1, nrhs, -one, a.Off(k+2-1, k-1).Vector(), 1, b.Off(k-1, 0).Vector(), b.Rows); err != nil {
					panic(err)
				}
				if err = b.Off(k+2-1, 0).Ger(n-k-1, nrhs, -one, a.Off(k+2-1, k).Vector(), 1, b.Off(k, 0).Vector(), b.Rows); err != nil {
					panic(err)
				}
			}

			//           Multiply by the inverse of the diagonal block.
			akm1k = a.Get(k, k-1)
			akm1 = a.Get(k-1, k-1) / akm1k
			ak = a.Get(k, k) / akm1k
			denom = akm1*ak - one
			for j = 1; j <= nrhs; j++ {
				bkm1 = b.Get(k-1, j-1) / akm1k
				bk = b.Get(k, j-1) / akm1k
				b.Set(k-1, j-1, (ak*bkm1-bk)/denom)
				b.Set(k, j-1, (akm1*bk-bkm1)/denom)
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
		k = n
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
			if k < n {
				if err = b.Off(k-1, 0).Vector().Gemv(Trans, n-k, nrhs, -one, b.Off(k, 0), a.Off(k, k-1).Vector(), 1, one, b.Rows); err != nil {
					panic(err)
				}
			}

			//           Interchange rows K and IPIV(K).
			kp = (*ipiv)[k-1]
			if kp != k {
				b.Off(kp-1, 0).Vector().Swap(nrhs, b.Off(k-1, 0).Vector(), b.Rows, b.Rows)
			}
			k = k - 1
		} else {
			//           2 x 2 diagonal block
			//
			//           Multiply by inv(L**T(K-1)), where L(K-1) is the transformation
			//           stored in columns K-1 and K of A.
			if k < n {
				if err = b.Off(k-1, 0).Vector().Gemv(Trans, n-k, nrhs, -one, b.Off(k, 0), a.Off(k, k-1).Vector(), 1, one, b.Rows); err != nil {
					panic(err)
				}
				if err = b.Off(k-1-1, 0).Vector().Gemv(Trans, n-k, nrhs, -one, b.Off(k, 0), a.Off(k, k-1-1).Vector(), 1, one, b.Rows); err != nil {
					panic(err)
				}
			}

			//           Interchange rows K and -IPIV(K) THEN K-1 and -IPIV(K-1)
			kp = -(*ipiv)[k-1]
			if kp != k {
				b.Off(kp-1, 0).Vector().Swap(nrhs, b.Off(k-1, 0).Vector(), b.Rows, b.Rows)
			}

			kp = -(*ipiv)[k-1-1]
			if kp != k-1 {
				b.Off(kp-1, 0).Vector().Swap(nrhs, b.Off(k-1-1, 0).Vector(), b.Rows, b.Rows)
			}

			k = k - 2
		}

		goto label90
	label100:
	}

	return
}
