package golapack

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/mat"
)

// Dsyswapr applies an elementary permutation on the rows and the columns of
// a symmetric matrix.
func Dsyswapr(uplo byte, n *int, a *mat.Matrix, lda, i1, i2 *int) {
	var upper bool
	var tmp float64
	var i int

	upper = uplo == 'U'
	if upper {
		//         UPPER
		//         first swap
		//          - swap column I1 and I2 from I1 to I1-1
		goblas.Dswap(toPtr((*i1)-1), a.Vector(0, (*i1)-1), toPtr(1), a.Vector(0, (*i2)-1), toPtr(1))

		//          second swap :
		//          - swap A(I1,I1) and A(I2,I2)
		//          - swap row I1 from I1+1 to I2-1 with col I2 from I1+1 to I2-1
		tmp = a.Get((*i1)-1, (*i1)-1)
		a.Set((*i1)-1, (*i1)-1, a.Get((*i2)-1, (*i2)-1))
		a.Set((*i2)-1, (*i2)-1, tmp)
		//
		for i = 1; i <= (*i2)-(*i1)-1; i++ {
			tmp = a.Get((*i1)-1, (*i1)+i-1)
			a.Set((*i1)-1, (*i1)+i-1, a.Get((*i1)+i-1, (*i2)-1))
			a.Set((*i1)+i-1, (*i2)-1, tmp)
		}
		//
		//          third swap
		//          - swap row I1 and I2 from I2+1 to N
		for i = (*i2) + 1; i <= (*n); i++ {
			tmp = a.Get((*i1)-1, i-1)
			a.Set((*i1)-1, i-1, a.Get((*i2)-1, i-1))
			a.Set((*i2)-1, i-1, tmp)
		}
		//
	} else {
		//
		//         LOWER
		//         first swap
		//          - swap row I1 and I2 from I1 to I1-1
		goblas.Dswap(toPtr((*i1)-1), a.Vector((*i1)-1, 0), lda, a.Vector((*i2)-1, 0), lda)

		//         second swap :
		//          - swap A(I1,I1) and A(I2,I2)
		//          - swap col I1 from I1+1 to I2-1 with row I2 from I1+1 to I2-1
		tmp = a.Get((*i1)-1, (*i1)-1)
		a.Set((*i1)-1, (*i1)-1, a.Get((*i2)-1, (*i2)-1))
		a.Set((*i2)-1, (*i2)-1, tmp)
		//
		for i = 1; i <= (*i2)-(*i1)-1; i++ {
			tmp = a.Get((*i1)+i-1, (*i1)-1)
			a.Set((*i1)+i-1, (*i1)-1, a.Get((*i2)-1, (*i1)+i-1))
			a.Set((*i2)-1, (*i1)+i-1, tmp)
		}
		//
		//         third swap
		//          - swap col I1 and I2 from I2+1 to N
		for i = (*i2) + 1; i <= (*n); i++ {
			tmp = a.Get(i-1, (*i1)-1)
			a.Set(i-1, (*i1)-1, a.Get(i-1, (*i2)-1))
			a.Set(i-1, (*i2)-1, tmp)
		}
		//
	}
}
