package golapack

import (
	"math/cmplx"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/mat"
)

// Zheswapr applies an elementary permutation on the rows and the columns of
// a hermitian matrix.
func Zheswapr(uplo byte, n *int, a *mat.CMatrix, lda, i1, i2 *int) {
	var upper bool
	var tmp complex128
	var i int

	upper = uplo == 'U'
	if upper {
		//         UPPER
		//         first swap
		//          - swap column I1 and I2 from I1 to I1-1
		goblas.Zswap(toPtr((*i1)-1), a.CVector(0, (*i1)-1), func() *int { y := 1; return &y }(), a.CVector(0, (*i2)-1), func() *int { y := 1; return &y }())
		//
		//          second swap :
		//          - swap A(I1,I1) and A(I2,I2)
		//          - swap row I1 from I1+1 to I2-1 with col I2 from I1+1 to I2-1
		//          - swap A(I2,I1) and A(I1,I2)
		tmp = a.Get((*i1)-1, (*i1)-1)
		a.Set((*i1)-1, (*i1)-1, a.Get((*i2)-1, (*i2)-1))
		a.Set((*i2)-1, (*i2)-1, tmp)

		for i = 1; i <= (*i2)-(*i1)-1; i++ {
			tmp = a.Get((*i1)-1, (*i1)+i-1)
			a.Set((*i1)-1, (*i1)+i-1, a.GetConj((*i1)+i-1, (*i2)-1))
			a.Set((*i1)+i-1, (*i2)-1, cmplx.Conj(tmp))
		}

		a.Set((*i1)-1, (*i2)-1, a.GetConj((*i1)-1, (*i2)-1))

		//          third swap
		//          - swap row I1 and I2 from I2+1 to N
		for i = (*i2) + 1; i <= (*n); i++ {
			tmp = a.Get((*i1)-1, i-1)
			a.Set((*i1)-1, i-1, a.Get((*i2)-1, i-1))
			a.Set((*i2)-1, i-1, tmp)
		}

	} else {
		//         LOWER
		//         first swap
		//          - swap row I1 and I2 from 1 to I1-1
		goblas.Zswap(toPtr((*i1)-1), a.CVector((*i1)-1, 0), lda, a.CVector((*i2)-1, 0), lda)

		//         second swap :
		//          - swap A(I1,I1) and A(I2,I2)
		//          - swap col I1 from I1+1 to I2-1 with row I2 from I1+1 to I2-1
		//          - swap A(I2,I1) and A(I1,I2)
		tmp = a.Get((*i1)-1, (*i1)-1)
		a.Set((*i1)-1, (*i1)-1, a.Get((*i2)-1, (*i2)-1))
		a.Set((*i2)-1, (*i2)-1, tmp)

		for i = 1; i <= (*i2)-(*i1)-1; i++ {
			tmp = a.Get((*i1)+i-1, (*i1)-1)
			a.Set((*i1)+i-1, (*i1)-1, a.GetConj((*i2)-1, (*i1)+i-1))
			a.Set((*i2)-1, (*i1)+i-1, cmplx.Conj(tmp))
		}

		a.Set((*i2)-1, (*i1)-1, a.GetConj((*i2)-1, (*i1)-1))

		//         third swap
		//          - swap col I1 and I2 from I2+1 to N
		for i = (*i2) + 1; i <= (*n); i++ {
			tmp = a.Get(i-1, (*i1)-1)
			a.Set(i-1, (*i1)-1, a.Get(i-1, (*i2)-1))
			a.Set(i-1, (*i2)-1, tmp)
		}

	}
}
