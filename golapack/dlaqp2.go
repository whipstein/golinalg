package golapack

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/mat"
)

// Dlaqp2 computes a QR factorization with column pivoting of
// the block A(OFFSET+1:M,1:N).
// The block A(1:OFFSET,1:N) is accordingly pivoted, but not factorized.
func Dlaqp2(m, n, offset *int, a *mat.Matrix, lda *int, jpvt *[]int, tau, vn1, vn2, work *mat.Vector) {
	var aii, one, temp, temp2, tol3z, zero float64
	var i, itemp, j, mn, offpi, pvt int

	zero = 0.0
	one = 1.0

	mn = min((*m)-(*offset), *n)
	tol3z = math.Sqrt(Dlamch(Epsilon))

	//     Compute factorization.
	for i = 1; i <= mn; i++ {

		offpi = (*offset) + i

		//        Determine ith pivot column and swap if necessary.
		pvt = (i - 1) + goblas.Idamax((*n)-i+1, vn1.Off(i-1))

		if pvt != i {
			goblas.Dswap(*m, a.Vector(0, pvt-1, 1), a.Vector(0, i-1, 1))
			itemp = (*jpvt)[pvt-1]
			(*jpvt)[pvt-1] = (*jpvt)[i-1]
			(*jpvt)[i-1] = itemp
			vn1.Set(pvt-1, vn1.Get(i-1))
			vn2.Set(pvt-1, vn2.Get(i-1))
		}

		//        Generate elementary reflector H(i).
		if offpi < (*m) {
			Dlarfg(toPtr((*m)-offpi+1), a.GetPtr(offpi-1, i-1), a.Vector(offpi, i-1), func() *int { y := 1; return &y }(), tau.GetPtr(i-1))
		} else {
			Dlarfg(func() *int { y := 1; return &y }(), a.GetPtr((*m)-1, i-1), a.Vector((*m)-1, i-1), func() *int { y := 1; return &y }(), tau.GetPtr(i-1))
		}

		if i < (*n) {
			//           Apply H(i)**T to A(offset+i:m,i+1:n) from the left.
			aii = a.Get(offpi-1, i-1)
			a.Set(offpi-1, i-1, one)
			Dlarf('L', toPtr((*m)-offpi+1), toPtr((*n)-i), a.Vector(offpi-1, i-1), func() *int { y := 1; return &y }(), tau.GetPtr(i-1), a.Off(offpi-1, i), lda, work)
			a.Set(offpi-1, i-1, aii)
		}

		//        Update partial column norms.
		for j = i + 1; j <= (*n); j++ {
			if vn1.Get(j-1) != zero {
				//              NOTE: The following 4 lines follow from the analysis in
				//              Lapack Working Note 176.
				temp = one - math.Pow(math.Abs(a.Get(offpi-1, j-1))/vn1.Get(j-1), 2)
				temp = math.Max(temp, zero)
				temp2 = temp * math.Pow(vn1.Get(j-1)/vn2.Get(j-1), 2)
				if temp2 <= tol3z {
					if offpi < (*m) {
						vn1.Set(j-1, goblas.Dnrm2((*m)-offpi, a.Vector(offpi, j-1, 1)))
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
