package lin

import (
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// dqpt01 tests the QR-factorization with pivoting of a matrix A.  The
// array AF contains the (possibly partial) QR-factorization of A, where
// the upper triangle of AF(1:k,1:k) is a partial triangular factor,
// the entries below the diagonal in the first k columns are the
// Householder vectors, and the rest of AF contains a partially updated
// matrix.
//
// This function returns ||A*P - Q*R||/(||norm(A)||*eps*M)
func dqpt01(m, n, k int, a, af *mat.Matrix, tau *mat.Vector, jpvt []int, work *mat.Vector, lwork int) (dqpt01Return float64) {
	var norma, one, zero float64
	var i, j int
	var err error

	rwork := vf(1)

	zero = 0.0
	one = 1.0

	dqpt01Return = zero

	//     Test if there is enough workspace
	if lwork < m*n+n {
		gltest.Xerbla("dqpt01", 10)
		return
	}

	//     Quick return if possible
	if m <= 0 || n <= 0 {
		return
	}

	norma = golapack.Dlange('O', m, n, a, rwork)

	for j = 1; j <= k; j++ {
		for i = 1; i <= min(j, m); i++ {
			work.Set((j-1)*m+i-1, af.Get(i-1, j-1))
		}
		for i = j + 1; i <= m; i++ {
			work.Set((j-1)*m+i-1, zero)
		}
	}
	for j = k + 1; j <= n; j++ {
		work.Off((j-1)*m).Copy(m, af.Off(0, j-1).Vector(), 1, 1)
	}

	if err = golapack.Dormqr(Left, NoTrans, m, n, k, af, tau, work.Matrix(m, opts), work.Off(m*n), lwork-m*n); err != nil {
		panic(err)
	}

	for j = 1; j <= n; j++ {
		//        Compare i-th column of QR and jpvt(i)-th column of A
		work.Off((j-1)*m).Axpy(m, -one, a.Off(0, jpvt[j-1]-1).Vector(), 1, 1)
	}

	dqpt01Return = golapack.Dlange('O', m, n, work.Matrix(m, opts), rwork) / (float64(max(m, n)) * golapack.Dlamch(Epsilon))
	if norma != zero {
		dqpt01Return = dqpt01Return / norma
	}

	return
}
