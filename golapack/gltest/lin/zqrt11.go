package lin

import (
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zqrt11 computes the test ratio
//
//       || Q'*Q - I || / (eps * m)
//
// where the orthogonal matrix Q is represented as a product of
// elementary transformations.  Each transformation has the form
//
//    H(k) = I - tau(k) v(k) v(k)'
//
// where tau(k) is stored in TAU(k) and v(k) is an m-vector of the form
// [ 0 ... 0 1 x(k) ]', where x(k) is a vector of length m-k stored
// in A(k+1:m,k).
func Zqrt11(m, k *int, a *mat.CMatrix, lda *int, tau, work *mat.CVector, lwork *int) (zqrt11Return float64) {
	var one, zero float64
	var info, j int

	rdummy := vf(1)

	zero = 0.0
	one = 1.0

	zqrt11Return = zero

	//     Test for sufficient workspace
	if (*lwork) < (*m)*(*m)+(*m) {
		gltest.Xerbla([]byte("ZQRT11"), 7)
		return
	}

	//     Quick return if possible
	if (*m) <= 0 {
		return
	}

	golapack.Zlaset('F', m, m, toPtrc128(complex(zero, 0)), toPtrc128(complex(one, 0)), work.CMatrix(*m, opts), m)

	//     Form Q
	golapack.Zunm2r('L', 'N', m, m, k, a, lda, tau, work.CMatrix(*m, opts), m, work.Off((*m)*(*m)+1-1), &info)

	//     Form Q'*Q
	golapack.Zunm2r('L', 'C', m, m, k, a, lda, tau, work.CMatrix(*m, opts), m, work.Off((*m)*(*m)+1-1), &info)

	for j = 1; j <= (*m); j++ {
		work.Set((j-1)*(*m)+j-1, work.Get((j-1)*(*m)+j-1)-complex(one, 0))
	}

	zqrt11Return = golapack.Zlange('O', m, m, work.CMatrix(*m, opts), m, rdummy) / (float64(*m) * golapack.Dlamch(Epsilon))

	return
}
