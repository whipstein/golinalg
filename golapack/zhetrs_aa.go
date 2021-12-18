package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zhetrsaa solves a system of linear equations A*X = B with a complex
// hermitian matrix A using the factorization A = U**H*T*U or
// A = L*T*L**H computed by ZHETRF_AA.
func ZhetrsAa(uplo mat.MatUplo, n, nrhs int, a *mat.CMatrix, ipiv *[]int, b *mat.CMatrix, work *mat.CVector, lwork int) (info int, err error) {
	var lquery, upper bool
	var one complex128
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
		gltest.Xerbla2("ZhetrsAa", err)
		return
	} else if lquery {
		lwkopt = (3*n - 2)
		work.SetRe(0, float64(lwkopt))
		return
	}

	//     Quick return if possible
	if n == 0 || nrhs == 0 {
		return
	}

	if upper {
		//        Solve A*X = B, where A = U**H*T*U.
		//
		//        1) Forward substitution with U**H
		if n > 1 {
			//           Pivot, P**T * B -> B
			for k = 1; k <= n; k++ {
				kp = (*ipiv)[k-1]
				if kp != k {
					b.Off(kp-1, 0).CVector().Swap(nrhs, b.Off(k-1, 0).CVector(), b.Rows, b.Rows)
				}
			}

			//           Compute U**H \ B -> B    [ (U**H \P**T * B) ]
			if err = b.Off(1, 0).Trsm(Left, Upper, ConjTrans, Unit, n-1, nrhs, one, a.Off(0, 1)); err != nil {
				panic(err)
			}
		}

		//        2) Solve with triangular matrix T
		//
		//        Compute T \ B -> B   [ T \ (U**H \P**T * B) ]
		Zlacpy(Full, 1, n, a.Off(0, 0).UpdateRows(a.Rows+1), work.Off(n-1).CMatrix(1, opts))
		if n > 1 {
			Zlacpy(Full, 1, n-1, a.Off(0, 1).UpdateRows(a.Rows+1), work.Off(2*n-1).CMatrix(1, opts))
			Zlacpy(Full, 1, n-1, a.Off(0, 1).UpdateRows(a.Rows+1), work.CMatrix(1, opts))
			Zlacgv(n-1, work, 1)
		}
		if info, err = Zgtsv(n, nrhs, work, work.Off(n-1), work.Off(2*n-1), b); err != nil {
			panic(err)
		}

		//        3) Backward substitution with U
		if n > 1 {
			//           Compute U \ B -> B   [ U \ (T \ (U**H \P**T * B) ) ]
			if err = b.Off(1, 0).Trsm(Left, Upper, NoTrans, Unit, n-1, nrhs, one, a.Off(0, 1)); err != nil {
				panic(err)
			}

			//           Pivot, P * B  [ P * (U**H \ (T \ (U \P**T * B) )) ]
			for k = n; k >= 1; k-- {
				kp = (*ipiv)[k-1]
				if kp != k {
					b.Off(kp-1, 0).CVector().Swap(nrhs, b.Off(k-1, 0).CVector(), b.Rows, b.Rows)
				}
			}
		}

	} else {
		//        Solve A*X = B, where A = L*T*L**H.
		//
		//        1) Forward substitution with L
		if n > 1 {
			//           Pivot, P**T * B -> B
			for k = 1; k <= n; k++ {
				kp = (*ipiv)[k-1]
				if kp != k {
					b.Off(kp-1, 0).CVector().Swap(nrhs, b.Off(k-1, 0).CVector(), b.Rows, b.Rows)
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
		Zlacpy(Full, 1, n, a.Off(0, 0).UpdateRows(a.Rows+1), work.Off(n-1).CMatrix(1, opts))
		if n > 1 {
			Zlacpy(Full, 1, n-1, a.Off(1, 0).UpdateRows(a.Rows+1), work.CMatrix(1, opts))
			Zlacpy(Full, 1, n-1, a.Off(1, 0).UpdateRows(a.Rows+1), work.Off(2*n-1).CMatrix(1, opts))
			Zlacgv(n-1, work.Off(2*n-1), 1)
		}
		if info, err = Zgtsv(n, nrhs, work, work.Off(n-1), work.Off(2*n-1), b); err != nil {
			panic(err)
		}

		//        3) Backward substitution with L**H
		if n > 1 {
			//           Compute L**H \ B -> B   [ L**H \ (T \ (L \P**T * B) ) ]
			if err = b.Off(1, 0).Trsm(Left, Lower, ConjTrans, Unit, n-1, nrhs, one, a.Off(1, 0)); err != nil {
				panic(err)
			}

			//           Pivot, P * B  [ P * (L**H \ (T \ (L \P**T * B) )) ]
			for k = n; k >= 1; k-- {
				kp = (*ipiv)[k-1]
				if kp != k {
					b.Off(kp-1, 0).CVector().Swap(nrhs, b.Off(k-1, 0).CVector(), b.Rows, b.Rows)
				}
			}
		}

	}

	return
}
