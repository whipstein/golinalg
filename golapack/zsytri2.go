package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zsytri2 computes the inverse of a COMPLEX*16 symmetric indefinite matrix
// A using the factorization A = U*D*U**T or A = L*D*L**T computed by
// ZSYTRF. Zsytri2 sets the LEADING DIMENSION of the workspace
// before calling Zsytri2X that actually computes the inverse.
func Zsytri2(uplo mat.MatUplo, n int, a *mat.CMatrix, ipiv *[]int, work *mat.CVector, lwork int) (info int, err error) {
	var lquery, upper bool
	var minsize, nbmax int

	//     Test the input parameters.
	upper = uplo == Upper
	lquery = (lwork == -1)
	//     Get blocksize
	nbmax = Ilaenv(1, "Zsytri2", []byte{uplo.Byte()}, n, -1, -1, -1)
	if nbmax >= n {
		minsize = n
	} else {
		minsize = (n + nbmax + 1) * (nbmax + 3)
	}

	if !upper && uplo != Lower {
		err = fmt.Errorf("!upper && uplo != Lower: uplo=%s", uplo)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if a.Rows < max(1, n) {
		err = fmt.Errorf("a.Rows < max(1, n): a.Rows=%v, n=%v", a.Rows, n)
	} else if lwork < minsize && !lquery {
		err = fmt.Errorf("lwork < minsize && !lquery: lwork=%v, minsize=%v, lquery=%v", lwork, minsize, lquery)
	}

	//     Quick return if possible
	if err != nil {
		gltest.Xerbla2("Zsytri2", err)
		return
	} else if lquery {
		work.SetRe(0, float64(minsize))
		return
	}
	if n == 0 {
		return
	}
	if nbmax >= n {
		if info, err = Zsytri(uplo, n, a, ipiv, work); err != nil {
			panic(err)
		}
	} else {
		if info, err = Zsytri2x(uplo, n, a, ipiv, work.CMatrix(n+nbmax+1, opts), nbmax); err != nil {
			panic(err)
		}
	}

	return
}
