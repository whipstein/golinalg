package golapack

import "github.com/whipstein/golinalg/mat"

// Dlatrz factors the M-by-(M+L) real upper trapezoidal matrix
// [ A1 A2 ] = [ A(1:M,1:M) A(1:M,N-L+1:N) ] as ( R  0 ) * Z, by means
// of orthogonal transformations.  Z is an (M+L)-by-(M+L) orthogonal
// matrix and, R and A1 are M-by-M upper triangular matrices.
func Dlatrz(m, n, l int, a *mat.Matrix, tau, work *mat.Vector) {
	var zero float64
	var i int

	zero = 0.0

	//     Test the input arguments
	//
	//     Quick return if possible
	if m == 0 {
		return
	} else if m == n {
		for i = 1; i <= n; i++ {
			tau.Set(i-1, zero)
		}
		return
	}

	for i = m; i >= 1; i-- {
		//        Generate elementary reflector H(i) to annihilate
		//        [ A(i,i) A(i,n-l+1:n) ]
		*a.GetPtr(i-1, i-1), *tau.GetPtr(i - 1) = Dlarfg(l+1, a.Get(i-1, i-1), a.Vector(i-1, n-l))

		//        Apply H(i) to A(1:i-1,i:n) from the right
		Dlarz(Right, i-1, n-i+1, l, a.Off(i-1, n-l), tau.Get(i-1), a.Off(0, i-1), work)

	}
}
