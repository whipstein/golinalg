package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zupgtr generates a complex unitary matrix Q which is defined as the
// product of n-1 elementary reflectors H(i) of order n, as returned by
// ZHPTRD using packed storage:
//
// if UPLO = 'U', Q = H(n-1) . . . H(2) H(1),
//
// if UPLO = 'L', Q = H(1) H(2) . . . H(n-1).
func Zupgtr(uplo mat.MatUplo, n int, ap, tau *mat.CVector, q *mat.CMatrix, work *mat.CVector) (err error) {
	var upper bool
	var cone, czero complex128
	var i, ij, j int

	czero = (0.0 + 0.0*1i)
	cone = (1.0 + 0.0*1i)

	//     Test the input arguments
	upper = uplo == Upper
	if !upper && uplo != Lower {
		err = fmt.Errorf("!upper && uplo != Lower: uplo=%s", uplo)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if q.Rows < max(1, n) {
		err = fmt.Errorf("q.Rows < max(1, n): q.Rows=%v, n=%v", q.Rows, n)
	}
	if err != nil {
		gltest.Xerbla2("Zupgtr", err)
		return
	}

	//     Quick return if possible
	if n == 0 {
		return
	}

	if upper {
		//        Q was determined by a call to ZHPTRD with UPLO = 'U'
		//
		//        Unpack the vectors which define the elementary reflectors and
		//        set the last row and column of Q equal to those of the unit
		//        matrix
		ij = 2
		for j = 1; j <= n-1; j++ {
			for i = 1; i <= j-1; i++ {
				q.Set(i-1, j-1, ap.Get(ij-1))
				ij = ij + 1
			}
			ij = ij + 2
			q.Set(n-1, j-1, czero)
		}
		for i = 1; i <= n-1; i++ {
			q.Set(i-1, n-1, czero)
		}
		q.Set(n-1, n-1, cone)

		//        Generate Q(1:n-1,1:n-1)
		if err = Zung2l(n-1, n-1, n-1, q, tau, work); err != nil {
			panic(err)
		}

	} else {
		//        Q was determined by a call to ZHPTRD with UPLO = 'L'.
		//
		//        Unpack the vectors which define the elementary reflectors and
		//        set the first row and column of Q equal to those of the unit
		//        matrix
		q.Set(0, 0, cone)
		for i = 2; i <= n; i++ {
			q.Set(i-1, 0, czero)
		}
		ij = 3
		for j = 2; j <= n; j++ {
			q.Set(0, j-1, czero)
			for i = j + 1; i <= n; i++ {
				q.Set(i-1, j-1, ap.Get(ij-1))
				ij = ij + 1
			}
			ij = ij + 2
		}
		if n > 1 {
			//           Generate Q(2:n,2:n)
			if err = Zung2r(n-1, n-1, n-1, q.Off(1, 1), tau, work); err != nil {
				panic(err)
			}
		}
	}

	return
}
