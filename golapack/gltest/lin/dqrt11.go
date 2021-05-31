package lin

import (
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dqrt11 computes the test ratio
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
func Dqrt11(m, k *int, a *mat.Matrix, lda *int, tau, work *mat.Vector, lwork *int) (dqrt11Return float64) {
	var one, zero float64
	var info, j int

	rdummy := vf(1)

	zero = 0.0
	one = 1.0

	dqrt11Return = zero

	//     Test for sufficient workspace
	if (*lwork) < (*m)*(*m)+(*m) {
		gltest.Xerbla([]byte("DQRT11"), 7)
		return
	}

	//     Quick return if possible
	if (*m) <= 0 {
		return
	}

	golapack.Dlaset('F', m, m, &zero, &one, work.Matrix(*m, opts), m)

	//     Form Q
	golapack.Dorm2r('L', 'N', m, m, k, a, lda, tau, work.Matrix(*m, opts), m, work.Off((*m)*(*m)+1-1), &info)

	//     Form Q'*Q
	golapack.Dorm2r('L', 'T', m, m, k, a, lda, tau, work.Matrix(*m, opts), m, work.Off((*m)*(*m)+1-1), &info)

	for j = 1; j <= (*m); j++ {
		work.Set((j-1)*(*m)+j-1, work.Get((j-1)*(*m)+j-1)-one)
	}

	dqrt11Return = golapack.Dlange('O', m, m, work.Matrix(*m, opts), m, rdummy) / (float64(*m) * golapack.Dlamch(Epsilon))

	return
}
