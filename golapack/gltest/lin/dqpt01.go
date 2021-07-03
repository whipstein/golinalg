package lin

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dqpt01 tests the QR-factorization with pivoting of a matrix A.  The
// array AF contains the (possibly partial) QR-factorization of A, where
// the upper triangle of AF(1:k,1:k) is a partial triangular factor,
// the entries below the diagonal in the first k columns are the
// Householder vectors, and the rest of AF contains a partially updated
// matrix.
//
// This function returns ||A*P - Q*R||/(||norm(A)||*eps*M)
func Dqpt01(m, n, k *int, a, af *mat.Matrix, lda *int, tau *mat.Vector, jpvt *[]int, work *mat.Vector, lwork *int) (dqpt01Return float64) {
	var norma, one, zero float64
	var i, info, j int

	rwork := vf(1)

	zero = 0.0
	one = 1.0

	dqpt01Return = zero

	//     Test if there is enough workspace
	if (*lwork) < (*m)*(*n)+(*n) {
		gltest.Xerbla([]byte("DQPT01"), 10)
		return
	}

	//     Quick return if possible
	if (*m) <= 0 || (*n) <= 0 {
		return
	}

	norma = golapack.Dlange('O', m, n, a, lda, rwork)

	for j = 1; j <= (*k); j++ {
		for i = 1; i <= minint(j, *m); i++ {
			work.Set((j-1)*(*m)+i-1, af.Get(i-1, j-1))
		}
		for i = j + 1; i <= (*m); i++ {
			work.Set((j-1)*(*m)+i-1, zero)
		}
	}
	for j = (*k) + 1; j <= (*n); j++ {
		goblas.Dcopy(*m, af.Vector(0, j-1), 1, work.Off((j-1)*(*m)+1-1), 1)
	}

	golapack.Dormqr('L', 'N', m, n, k, af, lda, tau, work.Matrix(*m, opts), m, work.Off((*m)*(*n)+1-1), toPtr((*lwork)-(*m)*(*n)), &info)

	for j = 1; j <= (*n); j++ {
		//        Compare i-th column of QR and jpvt(i)-th column of A
		goblas.Daxpy(*m, -one, a.Vector(0, (*jpvt)[j-1]-1), 1, work.Off((j-1)*(*m)+1-1), 1)
	}

	dqpt01Return = golapack.Dlange('O', m, n, work.Matrix(*m, opts), m, rwork) / (float64(maxint(*m, *n)) * golapack.Dlamch(Epsilon))
	if norma != zero {
		dqpt01Return = dqpt01Return / norma
	}

	return
}
