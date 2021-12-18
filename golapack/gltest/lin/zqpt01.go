package lin

import (
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// zqpt01 tests the QR-factorization with pivoting of a matrix A.  The
// array AF contains the (possibly partial) QR-factorization of A, where
// the upper triangle of AF(1:k,1:k) is a partial triangular factor,
// the entries below the diagonal in the first k columns are the
// Householder vectors, and the rest of AF contains a partially updated
// matrix.
//
// This function returns ||A*P - Q*R||/(||norm(A)||*eps*M)
func zqpt01(m, n, k int, a, af *mat.CMatrix, tau *mat.CVector, jpvt *[]int, work *mat.CVector, lwork int) (zqpt01Return float64) {
	var norma, one, zero float64
	var i, j int
	var err error

	rwork := vf(1)

	zero = 0.0
	one = 1.0

	zqpt01Return = zero

	//     Test if there is enough workspace
	if lwork < m*n+n {
		gltest.Xerbla("zqpt01", 10)
		return
	}

	//     Quick return if possible
	if m <= 0 || n <= 0 {
		return
	}

	norma = golapack.Zlange('O', m, n, a, rwork)

	for j = 1; j <= k; j++ {
		for i = 1; i <= min(j, m); i++ {
			work.Set((j-1)*m+i-1, af.Get(i-1, j-1))
		}
		for i = j + 1; i <= m; i++ {
			work.SetRe((j-1)*m+i-1, zero)
		}
	}
	for j = k + 1; j <= n; j++ {
		work.Off((j-1)*m).Copy(m, af.Off(0, j-1).CVector(), 1, 1)
	}

	if err = golapack.Zunmqr(Left, NoTrans, m, n, k, af, tau, work.CMatrix(m, opts), work.Off(m*n), lwork-m*n); err != nil {
		panic(err)
	}

	for j = 1; j <= n; j++ {
		//        Compare i-th column of QR and jpvt(i)-th column of A
		work.Off((j-1)*m).Axpy(m, complex(-one, 0), a.Off(0, (*jpvt)[j-1]-1).CVector(), 1, 1)
	}

	zqpt01Return = golapack.Zlange('O', m, n, work.CMatrix(m, opts), rwork) / (float64(max(m, n)) * golapack.Dlamch(Epsilon))
	if norma != zero {
		zqpt01Return = zqpt01Return / norma
	}

	return
}
