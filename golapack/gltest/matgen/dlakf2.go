package matgen

import (
	"golinalg/golapack"
	"golinalg/mat"
)

// Dlakf2 Form the 2*M*N by 2*M*N matrix
//
//        Z = [ kron(In, A)  -kron(B', Im) ]
//            [ kron(In, D)  -kron(E', Im) ],
//
// where In is the identity matrix of size n and X' is the transpose
// of X. kron(X, Y) is the Kronecker product between the matrices X
// and Y.
func Dlakf2(m, n *int, a *mat.Matrix, lda *int, b, d, e, z *mat.Matrix, ldz *int) {
	var zero float64
	var i, ik, j, jk, l, mn, mn2 int

	zero = 0.0

	//     Initialize Z
	mn = (*m) * (*n)
	mn2 = 2 * mn
	golapack.Dlaset('F', &mn2, &mn2, &zero, &zero, z, ldz)

	ik = 1
	for l = 1; l <= (*n); l++ {
		//        form kron(In, A)
		for i = 1; i <= (*m); i++ {
			for j = 1; j <= (*m); j++ {
				z.Set(ik+i-1-1, ik+j-1-1, a.Get(i-1, j-1))
			}
		}

		//        form kron(In, D)
		for i = 1; i <= (*m); i++ {
			for j = 1; j <= (*m); j++ {
				z.Set(ik+mn+i-1-1, ik+j-1-1, d.Get(i-1, j-1))
			}
		}

		ik = ik + (*m)
	}

	ik = 1
	for l = 1; l <= (*n); l++ {
		jk = mn + 1

		for j = 1; j <= (*n); j++ {
			//           form -kron(B', Im)
			for i = 1; i <= (*m); i++ {
				z.Set(ik+i-1-1, jk+i-1-1, -b.Get(j-1, l-1))
			}

			//           form -kron(E', Im)
			for i = 1; i <= (*m); i++ {
				z.Set(ik+mn+i-1-1, jk+i-1-1, -e.Get(j-1, l-1))
			}

			jk = jk + (*m)
		}

		ik = ik + (*m)
	}
}
