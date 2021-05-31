package lin

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zqpt01 tests the QR-factorization with pivoting of a matrix A.  The
// array AF contains the (possibly partial) QR-factorization of A, where
// the upper triangle of AF(1:k,1:k) is a partial triangular factor,
// the entries below the diagonal in the first k columns are the
// Householder vectors, and the rest of AF contains a partially updated
// matrix.
//
// This function returns ||A*P - Q*R||/(||norm(A)||*eps*M)
func Zqpt01(m, n, k *int, a, af *mat.CMatrix, lda *int, tau *mat.CVector, jpvt *[]int, work *mat.CVector, lwork *int) (zqpt01Return float64) {
	var norma, one, zero float64
	var i, info, j int

	rwork := vf(1)

	zero = 0.0
	one = 1.0

	zqpt01Return = zero

	//     Test if there is enough workspace
	if (*lwork) < (*m)*(*n)+(*n) {
		gltest.Xerbla([]byte("ZQPT01"), 10)
		return
	}

	//     Quick return if possible
	if (*m) <= 0 || (*n) <= 0 {
		return
	}

	norma = golapack.Zlange('O', m, n, a, lda, rwork)

	for j = 1; j <= (*k); j++ {
		for i = 1; i <= minint(j, *m); i++ {
			work.Set((j-1)*(*m)+i-1, af.Get(i-1, j-1))
		}
		for i = j + 1; i <= (*m); i++ {
			work.SetRe((j-1)*(*m)+i-1, zero)
		}
	}
	for j = (*k) + 1; j <= (*n); j++ {
		goblas.Zcopy(m, af.CVector(0, j-1), func() *int { y := 1; return &y }(), work.Off((j-1)*(*m)+1-1), func() *int { y := 1; return &y }())
	}

	golapack.Zunmqr('L', 'N', m, n, k, af, lda, tau, work.CMatrix(*m, opts), m, work.Off((*m)*(*n)+1-1), toPtr((*lwork)-(*m)*(*n)), &info)

	for j = 1; j <= (*n); j++ {
		//        Compare i-th column of QR and jpvt(i)-th column of A
		goblas.Zaxpy(m, toPtrc128(complex(-one, 0)), a.CVector(0, (*jpvt)[j-1]-1), func() *int { y := 1; return &y }(), work.Off((j-1)*(*m)+1-1), func() *int { y := 1; return &y }())
	}

	zqpt01Return = golapack.Zlange('O', m, n, work.CMatrix(*m, opts), m, rwork) / (float64(maxint(*m, *n)) * golapack.Dlamch(Epsilon))
	if norma != zero {
		zqpt01Return = zqpt01Return / norma
	}

	return
}
