package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// DsytrsAa solves a system of linear equations A*X = B with a real
// symmetric matrix A using the factorization A = U**T*T*U or
// A = L*T*L**T computed by DSYTRF_AA.
func DsytrsAa(uplo mat.MatUplo, n, nrhs int, a *mat.Matrix, ipiv *[]int, b *mat.Matrix, work *mat.Vector, lwork int) (info int, err error) {
	var lquery, upper bool
	var one float64
	var k, kp, lwkopt int

	one = 1.0

	upper = uplo == Upper
	lquery = (lwork == -1)
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
	} else if lwork < max(1, 3*n-2) && !lquery {
		err = fmt.Errorf("lwork < max(1, 3*n-2) && !lquery: lwork=%v, n=%v, lquery=%v", lwork, n, lquery)
	}
	if err != nil {
		gltest.Xerbla2("DsytrsAa", err)
		return
	} else if lquery {
		lwkopt = (3*n - 2)
		work.Set(0, float64(lwkopt))
		return
	}

	//     Quick return if possible
	if n == 0 || nrhs == 0 {
		return
	}

	if upper {
		//        Solve A*X = B, where A = U**T*T*U.
		//
		//        1) Forward substitution with U**T
		if n > 1 {
			//           Pivot, P**T * B -> B
			for k = 1; k <= n; k++ {
				kp = (*ipiv)[k-1]
				if kp != k {
					b.Off(kp-1, 0).Vector().Swap(nrhs, b.Off(k-1, 0).Vector(), b.Rows, b.Rows)
				}
			}

			//           Compute U**T \ B -> B    [ (U**T \P**T * B) ]
			if err = b.Off(1, 0).Trsm(Left, Upper, Trans, Unit, n-1, nrhs, one, a.Off(0, 1)); err != nil {
				panic(err)
			}
		}

		//        2) Solve with triangular matrix T
		//
		//        Compute T \ B -> B   [ T \ (U**T \P**T * B) ]
		Dlacpy(Full, 1, n, a.Off(0, 0).UpdateRows(a.Rows+1), work.Off(n-1).Matrix(1, opts))
		if n > 1 {
			Dlacpy(Full, 1, n-1, a.Off(0, 1).UpdateRows(a.Rows+1), work.Matrix(1, opts))
			Dlacpy(Full, 1, n-1, a.Off(0, 1).UpdateRows(a.Rows+1), work.Off(2*n-1).Matrix(1, opts))
		}
		if info, err = Dgtsv(n, nrhs, work, work.Off(n-1), work.Off(2*n-1), b); err != nil {
			panic(err)
		}

		//        3) Backward substitution with U
		if n > 1 {
			//           Compute U \ B -> B   [ U \ (T \ (U**T \P**T * B) ) ]
			if err = b.Off(1, 0).Trsm(Left, Upper, NoTrans, Unit, n-1, nrhs, one, a.Off(0, 1)); err != nil {
				panic(err)
			}

			//           Pivot, P * B -> B  [ P * (U \ (T \ (U**T \P**T * B) )) ]
			for k = n; k >= 1; k-- {
				kp = (*ipiv)[k-1]
				if kp != k {
					b.Off(kp-1, 0).Vector().Swap(nrhs, b.Off(k-1, 0).Vector(), b.Rows, b.Rows)
				}
			}
		}

	} else {
		//        Solve A*X = B, where A = L*T*L**T.
		//
		//        1) Forward substitution with L
		if n > 1 {
			//           Pivot, P**T * B -> B
			for k = 1; k <= n; k++ {
				kp = (*ipiv)[k-1]
				if kp != k {
					b.Off(kp-1, 0).Vector().Swap(nrhs, b.Off(k-1, 0).Vector(), b.Rows, b.Rows)
				}
			}

			//           Compute L \ B -> B    [ (L \P**T * B) ]
			if err = b.Off(1, 0).Trsm(Left, Lower, NoTrans, Unit, n-1, nrhs, one, a.Off(1, 0)); err != nil {
				panic(err)
			}
		}

		//        2) Solve with triangular matrix T
		//
		//        Compute T \ B -> B   [ T \ (L \P**T * B) ]
		Dlacpy(Full, 1, n, a.Off(0, 0).UpdateRows(a.Rows+1), work.Off(n-1).Matrix(1, opts))
		if n > 1 {
			Dlacpy(Full, 1, n-1, a.Off(1, 0).UpdateRows(a.Rows+1), work.Matrix(1, opts))
			Dlacpy(Full, 1, n-1, a.Off(1, 0).UpdateRows(a.Rows+1), work.Off(2*n-1).Matrix(1, opts))
		}
		if info, err = Dgtsv(n, nrhs, work, work.Off(n-1), work.Off(2*n-1), b); err != nil {
			panic(err)
		}

		//        3) Backward substitution with L**T
		if n > 1 {
			//           Compute (L**T \ B) -> B   [ L**T \ (T \ (L \P**T * B) ) ]
			if err = b.Off(1, 0).Trsm(Left, Lower, Trans, Unit, n-1, nrhs, one, a.Off(1, 0)); err != nil {
				panic(err)
			}

			//           Pivot, P * B -> B  [ P * (L**T \ (T \ (L \P**T * B) )) ]
			for k = n; k >= 1; k-- {
				kp = (*ipiv)[k-1]
				if kp != k {
					b.Off(kp-1, 0).Vector().Swap(nrhs, b.Off(k-1, 0).Vector(), b.Rows, b.Rows)
				}
			}
		}

	}

	return
}
