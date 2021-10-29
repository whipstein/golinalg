package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zungtr generates a complex unitary matrix Q which is defined as the
// product of n-1 elementary reflectors of order N, as returned by
// ZHETRD:
//
// if UPLO = 'U', Q = H(n-1) . . . H(2) H(1),
//
// if UPLO = 'L', Q = H(1) H(2) . . . H(n-1).
func Zungtr(uplo mat.MatUplo, n int, a *mat.CMatrix, tau, work *mat.CVector, lwork int) (err error) {
	var lquery, upper bool
	var one, zero complex128
	var i, j, lwkopt, nb int

	zero = (0.0 + 0.0*1i)
	one = (1.0 + 0.0*1i)

	//     Test the input arguments
	lquery = (lwork == -1)
	upper = uplo == Upper
	if !upper && uplo != Lower {
		err = fmt.Errorf("!upper && uplo != Lower: uplo=%s", uplo)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if a.Rows < max(1, n) {
		err = fmt.Errorf("a.Rows < max(1, n): a.Rows=%v, n=%v", a.Rows, n)
	} else if lwork < max(1, n-1) && !lquery {
		err = fmt.Errorf("lwork < max(1, n-1) && !lquery: lwork=%v, n=%v, lquery=%v", lwork, n, lquery)
	}

	if err == nil {
		if upper {
			nb = Ilaenv(1, "Zungql", []byte{' '}, n-1, n-1, n-1, -1)
		} else {
			nb = Ilaenv(1, "Zungqr", []byte{' '}, n-1, n-1, n-1, -1)
		}
		lwkopt = max(1, n-1) * nb
		work.SetRe(0, float64(lwkopt))
	}

	if err != nil {
		gltest.Xerbla2("Zungtr", err)
		return
	} else if lquery {
		return
	}

	//     Quick return if possible
	if n == 0 {
		work.Set(0, 1)
		return
	}

	if upper {
		//        Q was determined by a call to ZHETRD with UPLO = 'U'
		//
		//        Shift the vectors which define the elementary reflectors one
		//        column to the left, and set the last row and column of Q to
		//        those of the unit matrix
		for j = 1; j <= n-1; j++ {
			for i = 1; i <= j-1; i++ {
				a.Set(i-1, j-1, a.Get(i-1, j))
			}
			a.Set(n-1, j-1, zero)
		}
		for i = 1; i <= n-1; i++ {
			a.Set(i-1, n-1, zero)
		}
		a.Set(n-1, n-1, one)

		//        Generate Q(1:n-1,1:n-1)
		if err = Zungql(n-1, n-1, n-1, a, tau, work, lwork); err != nil {
			panic(err)
		}

	} else {
		//        Q was determined by a call to ZHETRD with UPLO = 'L'.
		//
		//        Shift the vectors which define the elementary reflectors one
		//        column to the right, and set the first row and column of Q to
		//        those of the unit matrix
		for j = n; j >= 2; j-- {
			a.Set(0, j-1, zero)
			for i = j + 1; i <= n; i++ {
				a.Set(i-1, j-1, a.Get(i-1, j-1-1))
			}
		}
		a.Set(0, 0, one)
		for i = 2; i <= n; i++ {
			a.Set(i-1, 0, zero)
		}
		if n > 1 {
			//           Generate Q(2:n,2:n)
			if err = Zungqr(n-1, n-1, n-1, a.Off(1, 1), tau, work, lwork); err != nil {
				panic(err)
			}
		}
	}
	work.SetRe(0, float64(lwkopt))

	return
}
