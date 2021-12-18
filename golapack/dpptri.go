package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dpptri computes the inverse of a real symmetric positive definite
// matrix A using the Cholesky factorization A = U**T*U or A = L*L**T
// computed by DPPTRF.
func Dpptri(uplo mat.MatUplo, n int, ap *mat.Vector) (info int, err error) {
	var upper bool
	var ajj, one float64
	var j, jc, jj, jjn int

	one = 1.0

	//     Test the input parameters.
	upper = uplo == Upper
	if !upper && uplo != Lower {
		err = fmt.Errorf("!upper && uplo != Lower: uplo=%s", uplo)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	}
	if err != nil {
		gltest.Xerbla2("Dpptri", err)
		return
	}

	//     Quick return if possible
	if n == 0 {
		return
	}

	//     Invert the triangular Cholesky factor U or L.
	if info, err = Dtptri(uplo, NonUnit, n, ap); err != nil {
		panic(err)
	}
	if info > 0 {
		return
	}

	if upper {
		//        Compute the product inv(U) * inv(U)**T.
		jj = 0
		for j = 1; j <= n; j++ {
			jc = jj + 1
			jj = jj + j
			if j > 1 {
				if err = ap.Spr(Upper, j-1, one, ap.Off(jc-1), 1); err != nil {
					panic(err)
				}
			}
			ajj = ap.Get(jj - 1)
			ap.Off(jc-1).Scal(j, ajj, 1)
		}

	} else {
		//        Compute the product inv(L)**T * inv(L).
		jj = 1
		for j = 1; j <= n; j++ {
			jjn = jj + n - j + 1
			ap.Set(jj-1, ap.Off(jj-1).Dot(n-j+1, ap.Off(jj-1), 1, 1))
			if j < n {
				if err = ap.Off(jj).Tpmv(Lower, Trans, NonUnit, n-j, ap.Off(jjn-1), 1); err != nil {
					panic(err)
				}
			}
			jj = jjn
		}
	}

	return
}
