package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dorbdb5 orthogonalizes the column vector
//      X = [ X1 ]
//          [ X2 ]
// with respect to the columns of
//      Q = [ Q1 ] .
//          [ Q2 ]
// The columns of Q must be orthonormal.
//
// If the projection is zero according to Kahan's "twice is enough"
// criterion, then some other vector from the orthogonal complement
// is returned. This vector is chosen in an arbitrary but deterministic
// way.
func Dorbdb5(m1, m2, n int, x1 *mat.Vector, incx1 int, x2 *mat.Vector, incx2 int, q1, q2 *mat.Matrix, work *mat.Vector, lwork int) (err error) {
	var one, zero float64
	var i, j int

	one = 1.0
	zero = 0.0

	//     Test input arguments
	if m1 < 0 {
		err = fmt.Errorf("m1 < 0: m1=%v", m1)
	} else if m2 < 0 {
		err = fmt.Errorf("m2 < 0: m2=%v", m2)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if incx1 < 1 {
		err = fmt.Errorf("x1.Inc < 1: x1.Inc=%v", incx1)
	} else if incx2 < 1 {
		err = fmt.Errorf("x2.Inc < 1: x2.Inc=%v", incx2)
	} else if q1.Rows < max(1, m1) {
		err = fmt.Errorf("q1.Rows < max(1, m1): q1.Rows=%v, m1=%v", q1.Rows, m1)
	} else if q2.Rows < max(1, m2) {
		err = fmt.Errorf("q2.Rows < max(1, m2): q2.Rows=%v, m2=%v", q2.Rows, m2)
	} else if lwork < n {
		err = fmt.Errorf("lwork < n: lwork=%v, n=%v", lwork, n)
	}

	if err != nil {
		gltest.Xerbla2("Dorbdb5", err)
		return
	}

	//     Project X onto the orthogonal complement of Q
	if err = Dorbdb6(m1, m2, n, x1, incx1, x2, incx2, q1, q2, work, lwork); err != nil {
		panic(err)
	}

	//     If the projection is nonzero, then return
	if x1.Nrm2(m1, incx1) != zero || x2.Nrm2(m2, incx2) != zero {
		return
	}

	//     Project each standard basis vector e_1,...,e_M1 in turn, stopping
	//     when a nonzero projection is found
	for i = 1; i <= m1; i++ {
		for j = 1; j <= m1; j++ {
			x1.Set(j-1, zero)
		}
		x1.Set(i-1, one)
		for j = 1; j <= m2; j++ {
			x2.Set(j-1, zero)
		}
		if err = Dorbdb6(m1, m2, n, x1, incx1, x2, incx2, q1, q2, work, lwork); err != nil {
			panic(err)
		}
		if x1.Nrm2(m1, incx1) != zero || x2.Nrm2(m2, incx2) != zero {
			return
		}
	}

	//     Project each standard basis vector e_(M1+1),...,e_(M1+M2) in turn,
	//     stopping when a nonzero projection is found
	for i = 1; i <= m2; i++ {
		for j = 1; j <= m1; j++ {
			x1.Set(j-1, zero)
		}
		for j = 1; j <= m2; j++ {
			x2.Set(j-1, zero)
		}
		x2.Set(i-1, one)
		if err = Dorbdb6(m1, m2, n, x1, incx1, x2, incx2, q1, q2, work, lwork); err != nil {
			panic(err)
		}
		if x1.Nrm2(m1, incx1) != zero || x2.Nrm2(m2, incx2) != zero {
			return
		}
	}

	return
}
