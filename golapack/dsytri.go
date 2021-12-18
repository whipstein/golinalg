package golapack

import (
	"fmt"
	"math"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dsytri computes the inverse of a real symmetric indefinite matrix
// A using the factorization A = U*D*U**T or A = L*D*L**T computed by
// DSYTRF.
func Dsytri(uplo mat.MatUplo, n int, a *mat.Matrix, ipiv *[]int, work *mat.Vector) (info int, err error) {
	var upper bool
	var ak, akkp1, akp1, d, one, t, temp, zero float64
	var k, kp, kstep int

	one = 1.0
	zero = 0.0

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
		gltest.Xerbla2("Dsytri", err)
		return
	}

	//     Quick return if possible
	if n == 0 {
		return
	}

	//     Check that the diagonal matrix D is nonsingular.
	if upper {
		//        Upper triangular storage: examine D from bottom to top
		for info = n; info >= 1; info-- {
			if (*ipiv)[info-1] > 0 && a.Get(info-1, info-1) == zero {
				return
			}
		}
	} else {
		//        Lower triangular storage: examine D from top to bottom.
		for info = 1; info <= n; info++ {
			if (*ipiv)[info-1] > 0 && a.Get(info-1, info-1) == zero {
				return
			}
		}
	}
	info = 0

	if upper {
		//        Compute inv(A) from the factorization A = U*D*U**T.
		//
		//        K is the main loop index, increasing from 1 to N in steps of
		//        1 or 2, depending on the size of the diagonal blocks.
		k = 1
	label30:
		;

		//        If K > N, exit from loop.
		if k > n {
			goto label40
		}

		if (*ipiv)[k-1] > 0 {
			//           1 x 1 diagonal block
			//
			//           Invert the diagonal block.
			a.Set(k-1, k-1, one/a.Get(k-1, k-1))
			//           Compute column K of the inverse.
			if k > 1 {
				work.Copy(k-1, a.Off(0, k-1).Vector(), 1, 1)
				err = a.Off(0, k-1).Vector().Symv(uplo, k-1, one, a, work, 1, zero, 1)
				a.Set(k-1, k-1, a.Get(k-1, k-1)-a.Off(0, k-1).Vector().Dot(k-1, work, 1, 1))
			}
			kstep = 1
		} else {
			//           2 x 2 diagonal block
			//
			//           Invert the diagonal block.
			t = math.Abs(a.Get(k-1, k))
			ak = a.Get(k-1, k-1) / t
			akp1 = a.Get(k, k) / t
			akkp1 = a.Get(k-1, k) / t
			d = t * (ak*akp1 - one)
			a.Set(k-1, k-1, akp1/d)
			a.Set(k, k, ak/d)
			a.Set(k-1, k, -akkp1/d)

			//           Compute columns K and K+1 of the inverse.
			if k > 1 {
				work.Copy(k-1, a.Off(0, k-1).Vector(), 1, 1)
				err = a.Off(0, k-1).Vector().Symv(uplo, k-1, one, a, work, 1, zero, 1)
				a.Set(k-1, k-1, a.Get(k-1, k-1)-a.Off(0, k-1).Vector().Dot(k-1, work, 1, 1))
				a.Set(k-1, k, a.Get(k-1, k)-a.Off(0, k).Vector().Dot(k-1, a.Off(0, k-1).Vector(), 1, 1))
				work.Copy(k-1, a.Off(0, k).Vector(), 1, 1)
				err = a.Off(0, k).Vector().Symv(uplo, k-1, one, a, work, 1, zero, 1)
				a.Set(k, k, a.Get(k, k)-a.Off(0, k).Vector().Dot(k-1, work, 1, 1))
			}
			kstep = 2
		}

		kp = abs((*ipiv)[k-1])
		if kp != k {
			//           Interchange rows and columns K and KP in the leading
			//           submatrix A(1:k+1,1:k+1)
			a.Off(0, kp-1).Vector().Swap(kp-1, a.Off(0, k-1).Vector(), 1, 1)
			a.Off(kp-1, kp).Vector().Swap(k-kp-1, a.Off(kp, k-1).Vector(), 1, a.Rows)
			temp = a.Get(k-1, k-1)
			a.Set(k-1, k-1, a.Get(kp-1, kp-1))
			a.Set(kp-1, kp-1, temp)
			if kstep == 2 {
				temp = a.Get(k-1, k)
				a.Set(k-1, k, a.Get(kp-1, k))
				a.Set(kp-1, k, temp)
			}
		}

		k = k + kstep
		goto label30
	label40:
	} else {
		//        Compute inv(A) from the factorization A = L*D*L**T.
		//
		//        K is the main loop index, increasing from 1 to N in steps of
		//        1 or 2, depending on the size of the diagonal blocks.
		k = n
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
			a.Set(k-1, k-1, one/a.Get(k-1, k-1))

			//           Compute column K of the inverse.
			if k < n {
				work.Copy(n-k, a.Off(k, k-1).Vector(), 1, 1)
				err = a.Off(k, k-1).Vector().Symv(uplo, n-k, one, a.Off(k, k), work, 1, zero, 1)
				a.Set(k-1, k-1, a.Get(k-1, k-1)-a.Off(k, k-1).Vector().Dot(n-k, work, 1, 1))
			}
			kstep = 1
		} else {
			//           2 x 2 diagonal block
			//
			//           Invert the diagonal block.
			t = math.Abs(a.Get(k-1, k-1-1))
			ak = a.Get(k-1-1, k-1-1) / t
			akp1 = a.Get(k-1, k-1) / t
			akkp1 = a.Get(k-1, k-1-1) / t
			d = t * (ak*akp1 - one)
			a.Set(k-1-1, k-1-1, akp1/d)
			a.Set(k-1, k-1, ak/d)
			a.Set(k-1, k-1-1, -akkp1/d)

			//           Compute columns K-1 and K of the inverse.
			if k < n {
				work.Copy(n-k, a.Off(k, k-1).Vector(), 1, 1)
				err = a.Off(k, k-1).Vector().Symv(uplo, n-k, one, a.Off(k, k), work, 1, zero, 1)
				a.Set(k-1, k-1, a.Get(k-1, k-1)-a.Off(k, k-1).Vector().Dot(n-k, work, 1, 1))
				a.Set(k-1, k-1-1, a.Get(k-1, k-1-1)-a.Off(k, k-1-1).Vector().Dot(n-k, a.Off(k, k-1).Vector(), 1, 1))
				work.Copy(n-k, a.Off(k, k-1-1).Vector(), 1, 1)
				err = a.Off(k, k-1-1).Vector().Symv(uplo, n-k, one, a.Off(k, k), work, 1, zero, 1)
				a.Set(k-1-1, k-1-1, a.Get(k-1-1, k-1-1)-a.Off(k, k-1-1).Vector().Dot(n-k, work, 1, 1))
			}
			kstep = 2
		}

		kp = abs((*ipiv)[k-1])
		if kp != k {
			//           Interchange rows and columns K and KP in the trailing
			//           submatrix A(k-1:n,k-1:n)
			if kp < n {
				a.Off(kp, kp-1).Vector().Swap(n-kp, a.Off(kp, k-1).Vector(), 1, 1)
			}
			a.Off(kp-1, k).Vector().Swap(kp-k-1, a.Off(k, k-1).Vector(), 1, a.Rows)
			temp = a.Get(k-1, k-1)
			a.Set(k-1, k-1, a.Get(kp-1, kp-1))
			a.Set(kp-1, kp-1, temp)
			if kstep == 2 {
				temp = a.Get(k-1, k-1-1)
				a.Set(k-1, k-1-1, a.Get(kp-1, k-1-1))
				a.Set(kp-1, k-1-1, temp)
			}
		}

		k = k - kstep
		goto label50
	label60:
	}

	return
}
