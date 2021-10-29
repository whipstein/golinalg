package lin

import (
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// dqrt11 computes the test ratio
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
func dqrt11(m, k int, a *mat.Matrix, tau, work *mat.Vector, lwork int) (dqrt11Return float64) {
	var one, zero float64
	var j int
	var err error

	rdummy := vf(1)

	zero = 0.0
	one = 1.0

	dqrt11Return = zero

	//     Test for sufficient workspace
	if lwork < m*m+m {
		gltest.Xerbla("dqrt11", 7)
		return
	}

	//     Quick return if possible
	if m <= 0 {
		return
	}

	golapack.Dlaset(Full, m, m, zero, one, work.Matrix(m, opts))

	//     Form Q
	if err = golapack.Dorm2r(Left, NoTrans, m, m, k, a, tau, work.Matrix(m, opts), work.Off(m*m)); err != nil {
		panic(err)
	}

	//     Form Q'*Q
	if err = golapack.Dorm2r(Left, Trans, m, m, k, a, tau, work.Matrix(m, opts), work.Off(m*m)); err != nil {
		panic(err)
	}

	for j = 1; j <= m; j++ {
		work.Set((j-1)*m+j-1, work.Get((j-1)*m+j-1)-one)
	}

	dqrt11Return = golapack.Dlange('O', m, m, work.Matrix(m, opts), rdummy) / (float64(m) * golapack.Dlamch(Epsilon))

	return
}
