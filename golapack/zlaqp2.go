package golapack

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/mat"
)

// Zlaqp2 computes a QR factorization with column pivoting of
// the block A(OFFSET+1:M,1:N).
// The block A(1:OFFSET,1:N) is accordingly pivoted, but not factorized.
func Zlaqp2(m, n, offset *int, a *mat.CMatrix, lda *int, jpvt *[]int, tau *mat.CVector, vn1, vn2 *mat.Vector, work *mat.CVector) {
	var aii, cone complex128
	var one, temp, temp2, tol3z, zero float64
	var i, itemp, j, mn, offpi, pvt int

	zero = 0.0
	one = 1.0
	cone = (1.0 + 0.0*1i)

	mn = minint((*m)-(*offset), *n)
	tol3z = math.Sqrt(Dlamch(Epsilon))

	//     Compute factorization.
	for i = 1; i <= mn; i++ {

		offpi = (*offset) + i

		//        Determine ith pivot column and swap if necessary.
		pvt = (i - 1) + goblas.Idamax(toPtr((*n)-i+1), vn1.Off(i-1), func() *int { y := 1; return &y }())

		if pvt != i {
			goblas.Zswap(m, a.CVector(0, pvt-1), func() *int { y := 1; return &y }(), a.CVector(0, i-1), func() *int { y := 1; return &y }())
			itemp = (*jpvt)[pvt-1]
			(*jpvt)[pvt-1] = (*jpvt)[i-1]
			(*jpvt)[i-1] = itemp
			vn1.Set(pvt-1, vn1.Get(i-1))
			vn2.Set(pvt-1, vn2.Get(i-1))
		}

		//        Generate elementary reflector H(i).
		if offpi < (*m) {
			Zlarfg(toPtr((*m)-offpi+1), a.GetPtr(offpi-1, i-1), a.CVector(offpi+1-1, i-1), func() *int { y := 1; return &y }(), tau.GetPtr(i-1))
		} else {
			Zlarfg(func() *int { y := 1; return &y }(), a.GetPtr((*m)-1, i-1), a.CVector((*m)-1, i-1), func() *int { y := 1; return &y }(), tau.GetPtr(i-1))
		}

		if i < (*n) {
			//           Apply H(i)**H to A(offset+i:m,i+1:n) from the left.
			aii = a.Get(offpi-1, i-1)
			a.Set(offpi-1, i-1, cone)
			Zlarf('L', toPtr((*m)-offpi+1), toPtr((*n)-i), a.CVector(offpi-1, i-1), func() *int { y := 1; return &y }(), toPtrc128(tau.GetConj(i-1)), a.Off(offpi-1, i+1-1), lda, work)
			a.Set(offpi-1, i-1, aii)
		}

		//        Update partial column norms.
		for j = i + 1; j <= (*n); j++ {
			if vn1.Get(j-1) != zero {
				//              NOTE: The following 4 lines follow from the analysis in
				//              Lapack Working Note 176.
				temp = one - math.Pow(a.GetMag(offpi-1, j-1)/vn1.Get(j-1), 2)
				temp = maxf64(temp, zero)
				temp2 = temp * math.Pow(vn1.Get(j-1)/vn2.Get(j-1), 2)
				if temp2 <= tol3z {
					if offpi < (*m) {
						vn1.Set(j-1, goblas.Dznrm2(toPtr((*m)-offpi), a.CVector(offpi+1-1, j-1), func() *int { y := 1; return &y }()))
						vn2.Set(j-1, vn1.Get(j-1))
					} else {
						vn1.Set(j-1, zero)
						vn2.Set(j-1, zero)
					}
				} else {
					vn1.Set(j-1, vn1.Get(j-1)*math.Sqrt(temp))
				}
			}
		}

	}
}
