package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zsytrsaa2stage solves a system of linear equations A*X = B with a complex
// symmetric matrix A using the factorization A = U**T*T*U or
// A = L*T*L**T computed by ZSYTRF_AA_2STAGE.
func ZsytrsAa2stage(uplo mat.MatUplo, n, nrhs int, a *mat.CMatrix, tb *mat.CVector, ltb int, ipiv, ipiv2 *[]int, b *mat.CMatrix) (err error) {
	var upper bool
	var one complex128
	var ldtb, nb int

	one = (1.0 + 0.0*1i)

	upper = uplo == Upper
	if !upper && uplo != Lower {
		err = fmt.Errorf("!upper && uplo != Lower: uplo=%s", uplo)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if nrhs < 0 {
		err = fmt.Errorf("nrhs < 0: nrhs=%v", nrhs)
	} else if a.Rows < max(1, n) {
		err = fmt.Errorf("a.Rows < max(1, n): a.Rows=%v, n=%v", a.Rows, n)
	} else if ltb < (4 * n) {
		err = fmt.Errorf("ltb < (4 * n): ltb=%v, n=%v", ltb, n)
	} else if b.Rows < max(1, n) {
		err = fmt.Errorf("b.Rows < max(1, n): b.Rows=%v, n=%v", b.Rows, n)
	}
	if err != nil {
		gltest.Xerbla2("ZsytrsAa2stage", err)
		return
	}

	//     Quick return if possible
	if n == 0 || nrhs == 0 {
		return
	}

	//     Read NB and compute LDTB
	nb = int(tb.GetRe(0))
	ldtb = ltb / n

	if upper {
		//        Solve A*X = B, where A = U**T*T*U.
		if n > nb {
			//           Pivot, P**T * B -> B
			Zlaswp(nrhs, b, nb+1, n, ipiv, 1)

			//           Compute (U**T \ B) -> B    [ (U**T \P**T * B) ]
			if err = b.Off(nb, 0).Trsm(Left, Upper, Trans, Unit, n-nb, nrhs, one, a.Off(0, nb)); err != nil {
				panic(err)
			}

		}

		//        Compute T \ B -> B   [ T \ (U**T \P**T * B) ]
		if err = Zgbtrs(NoTrans, n, nb, nb, nrhs, tb.CMatrix(ldtb, opts), ipiv2, b); err != nil {
			panic(err)
		}
		if n > nb {
			//           Compute (U \ B) -> B   [ U \ (T \ (U**T \P**T * B) ) ]
			if err = b.Off(nb, 0).Trsm(Left, Upper, NoTrans, Unit, n-nb, nrhs, one, a.Off(0, nb)); err != nil {
				panic(err)
			}

			//           Pivot, P * B -> B  [ P * (U \ (T \ (U**T \P**T * B) )) ]
			Zlaswp(nrhs, b, nb+1, n, ipiv, -1)

		}

	} else {
		//        Solve A*X = B, where A = L*T*L**T.
		if n > nb {
			//           Pivot, P**T * B -> B
			Zlaswp(nrhs, b, nb+1, n, ipiv, 1)

			//           Compute (L \ B) -> B    [ (L \P**T * B) ]
			if err = b.Off(nb, 0).Trsm(Left, Lower, NoTrans, Unit, n-nb, nrhs, one, a.Off(nb, 0)); err != nil {
				panic(err)
			}

		}

		//        Compute T \ B -> B   [ T \ (L \P**T * B) ]
		if err = Zgbtrs(NoTrans, n, nb, nb, nrhs, tb.CMatrix(ldtb, opts), ipiv2, b); err != nil {
			panic(err)
		}
		if n > nb {
			//           Compute (L**T \ B) -> B   [ L**T \ (T \ (L \P**T * B) ) ]
			if err = b.Off(nb, 0).Trsm(Left, Lower, Trans, Unit, n-nb, nrhs, one, a.Off(nb, 0)); err != nil {
				panic(err)
			}

			//           Pivot, P * B -> B  [ P * (L**T \ (T \ (L \P**T * B) )) ]
			Zlaswp(nrhs, b, nb+1, n, ipiv, -1)

		}
	}

	return
}
