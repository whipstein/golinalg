package lin

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// zqrt11 computes the test ratio
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
func zqrt11(m, k int, a *mat.CMatrix, tau, work *mat.CVector, lwork int) (zqrt11Return float64) {
	var one, zero float64
	var j int
	var err error

	rdummy := vf(1)

	zero = 0.0
	one = 1.0

	zqrt11Return = zero

	//     Test for sufficient workspace
	if lwork < m*m+m {
		err = fmt.Errorf("lwork < m*m+m: lwork=%v, m=%v", lwork, m)
		gltest.Xerbla2("zqrt11", err)
		return
	}

	//     Quick return if possible
	if m <= 0 {
		return
	}

	golapack.Zlaset(Full, m, m, complex(zero, 0), complex(one, 0), work.CMatrix(m, opts))

	//     Form Q
	if err = golapack.Zunm2r(Left, NoTrans, m, m, k, a, tau, work.CMatrix(m, opts), work.Off(m*m)); err != nil {
		panic(err)
	}

	//     Form Q'*Q
	if err = golapack.Zunm2r(Left, ConjTrans, m, m, k, a, tau, work.CMatrix(m, opts), work.Off(m*m)); err != nil {
		panic(err)
	}

	for j = 1; j <= m; j++ {
		work.Set((j-1)*m+j-1, work.Get((j-1)*m+j-1)-complex(one, 0))
	}

	zqrt11Return = golapack.Zlange('O', m, m, work.CMatrix(m, opts), rdummy) / (float64(m) * golapack.Dlamch(Epsilon))

	return
}
