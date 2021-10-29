package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dsytri2 computes the inverse of a DOUBLE PRECISION symmetric indefinite matrix
// A using the factorization A = U*D*U**T or A = L*D*L**T computed by
// DSYTRF. Dsytri2 sets the LEADING DIMENSION of the workspace
// before calling Dsytri2X that actually computes the inverse.
func Dsytri2(uplo mat.MatUplo, n int, a *mat.Matrix, ipiv *[]int, work *mat.Matrix, lwork int) (info int, err error) {
	var lquery, upper bool
	var minsize, nbmax int

	//     Test the input parameters.
	upper = uplo == Upper
	lquery = (lwork == -1)
	//     Get blocksize
	nbmax = Ilaenv(1, "Dsytri2", []byte{uplo.Byte()}, n, -1, -1, -1)
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
		gltest.Xerbla2("Dsytri2", err)
		return
	} else if lquery {
		work.SetIdx(0, float64(minsize))
		return
	}
	if n == 0 {
		return
	}
	if nbmax >= n {
		if info, err = Dsytri(uplo, n, a, ipiv, work.VectorIdx(0)); err != nil {
			panic(err)
		}
	} else {
		if info, err = Dsytri2x(uplo, n, a, ipiv, work.VectorIdx(0), nbmax); err != nil {
			panic(err)
		}
	}

	return
}
