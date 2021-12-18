package lin

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// dgeqls Solve the least squares problem
//     min || A*X - B ||
// using the QL factorization
//     A = Q*L
// computed by DGEQLF.
func dgeqls(m, n, nrhs int, a *mat.Matrix, tau *mat.Vector, b *mat.Matrix, work *mat.Vector, lwork int) (err error) {
	var one float64

	one = 1.0

	//     Test the input arguments.
	if m < 0 {
		err = fmt.Errorf("m < 0: m=%v", m)
	} else if n < 0 || n > m {
		err = fmt.Errorf("n < 0 || n > m: m=%v, n=%v", m, n)
	} else if nrhs < 0 {
		err = fmt.Errorf("nrhs < 0: nrhs=%v", nrhs)
	} else if a.Rows < max(1, m) {
		err = fmt.Errorf("a.Rows < max(1, m): a.Rows=%v, m=%v", a.Rows, m)
	} else if b.Rows < max(1, m) {
		err = fmt.Errorf("b.Rows < max(1, m): b.Rows=%v, m=%v", b.Rows, m)
	} else if lwork < 1 || lwork < nrhs && m > 0 && n > 0 {
		err = fmt.Errorf("lwork < 1 || lwork < nrhs && m > 0 && n > 0: lwork=%v, nrhs=%v, m=%v, n=%v", lwork, nrhs, m, n)
	}
	if err != nil {
		gltest.Xerbla2("Dgeqls", err)
		return
	}

	//     Quick return if possible
	if n == 0 || nrhs == 0 || m == 0 {
		return
	}

	//     B := Q' * B
	if err = golapack.Dormql(Left, Trans, m, nrhs, n, a, tau, b, work, lwork); err != nil {
		panic(err)
	}

	//     Solve L*X = B(m-n+1:m,:)
	if err = b.Off(m-n, 0).Trsm(mat.Left, mat.Lower, mat.NoTrans, mat.NonUnit, n, nrhs, one, a.Off(m-n, 0)); err != nil {
		panic(err)
	}

	return
}
