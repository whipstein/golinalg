package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zunbdb5 orthogonalizes the column vector
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
func Zunbdb5(m1, m2, n int, x1, x2 *mat.CVector, q1, q2 *mat.CMatrix, work *mat.CVector, lwork int) (err error) {
	var one, zero complex128
	var i, j int

	one = (1.0 + 0.0*1i)
	zero = (0.0 + 0.0*1i)

	//     Test input arguments
	if m1 < 0 {
		err = fmt.Errorf("m1 < 0: m1=%v", m1)
	} else if m2 < 0 {
		err = fmt.Errorf("m2 < 0: m2=%v", m2)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if x1.Inc < 1 {
		err = fmt.Errorf("x1.Inc < 1: x1.Inc=%v", x1.Inc)
	} else if x2.Inc < 1 {
		err = fmt.Errorf("x2.Inc < 1: x2.Inc=%v", x2.Inc)
	} else if q1.Rows < max(1, m1) {
		err = fmt.Errorf("q1.Rows < max(1, m1): q1.Rows=%v, m1=%v", q1.Rows, m1)
	} else if q2.Rows < max(1, m2) {
		err = fmt.Errorf("q2.Rows < max(1, m2): q2.Rows=%v, m2=%v", q2.Rows, m2)
	} else if lwork < n {
		err = fmt.Errorf("lwork < n: lwork=%v", lwork)
	}

	if err != nil {
		gltest.Xerbla2("Zunbdb5", err)
		return
	}

	//     Project X onto the orthogonal complement of Q
	if err = Zunbdb6(m1, m2, n, x1, x2, q1, q2, work, lwork); err != nil {
		panic(err)
	}

	//     If the projection is nonzero, then return
	if goblas.Dznrm2(m1, x1) != real(zero) || goblas.Dznrm2(m2, x2) != real(zero) {
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
		if err = Zunbdb6(m1, m2, n, x1, x2, q1, q2, work, lwork); err != nil {
			panic(err)
		}
		if goblas.Dznrm2(m1, x1) != real(zero) || goblas.Dznrm2(m2, x2) != real(zero) {
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
		if err = Zunbdb6(m1, m2, n, x1, x2, q1, q2, work, lwork); err != nil {
			panic(err)
		}
		if goblas.Dznrm2(m1, x1) != real(zero) || goblas.Dznrm2(m2, x2) != real(zero) {
			return
		}
	}

	return
}
