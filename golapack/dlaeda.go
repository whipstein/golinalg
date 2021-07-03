package golapack

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dlaeda computes the Z vector corresponding to the merge step in the
// CURLVLth step of the merge process with TLVLS steps for the CURPBMth
// problem.
func Dlaeda(n, tlvls, curlvl, curpbm *int, prmptr, perm, givptr *[]int, givcol *[]int, givnum *mat.Matrix, q *mat.Vector, qptr *[]int, z, ztemp *mat.Vector, info *int) {
	var half, one, zero float64
	var bsiz1, bsiz2, curr, i, k, mid, psiz1, psiz2, ptr, zptr1 int
	var err error
	_ = err

	zero = 0.0
	half = 0.5
	one = 1.0

	//     Test the input parameters.
	(*info) = 0

	if (*n) < 0 {
		(*info) = -1
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("DLAEDA"), -(*info))
		return
	}

	//     Quick return if possible
	if (*n) == 0 {
		return
	}

	//     Determine location of first number in second half.
	mid = (*n)/2 + 1

	//     Gather last/first rows of appropriate eigenblocks into center of Z
	ptr = 1

	//     Determine location of lowest level subproblem in the full storage
	//     scheme
	curr = ptr + (*curpbm)*int(math.Pow(2, float64(*curlvl))) + int(math.Pow(2, float64((*curlvl)-1))) - 1

	//     Determine size of these matrices.  We add HALF to the value of
	//     the SQRT in case the machine underestimates one of these square
	//     roots.
	bsiz1 = int(half + math.Sqrt(float64((*qptr)[curr+1-1]-(*qptr)[curr-1])))
	bsiz2 = int(half + math.Sqrt(float64((*qptr)[curr+2-1]-(*qptr)[curr+1-1])))
	for k = 1; k <= mid-bsiz1-1; k++ {
		z.Set(k-1, zero)
	}
	goblas.Dcopy(bsiz1, q.Off((*qptr)[curr-1]+bsiz1-1-1), bsiz1, z.Off(mid-bsiz1-1), 1)
	goblas.Dcopy(bsiz2, q.Off((*qptr)[curr+1-1]-1), bsiz2, z.Off(mid-1), 1)
	for k = mid + bsiz2; k <= (*n); k++ {
		z.Set(k-1, zero)
	}

	//     Loop through remaining levels 1 -> CURLVL applying the Givens
	//     rotations and permutation and then multiplying the center matrices
	//     against the current Z.
	ptr = int(math.Pow(2, float64(*tlvls))) + 1
	for k = 1; k <= (*curlvl)-1; k++ {
		curr = ptr + (*curpbm)*int(math.Pow(2, float64((*curlvl)-k))) + int(math.Pow(2, float64((*curlvl)-k-1))) - 1
		psiz1 = (*prmptr)[curr+1-1] - (*prmptr)[curr-1]
		psiz2 = (*prmptr)[curr+2-1] - (*prmptr)[curr+1-1]
		zptr1 = mid - psiz1

		//       Apply Givens at CURR and CURR+1
		for i = (*givptr)[curr-1]; i <= (*givptr)[curr+1-1]-1; i++ {
			goblas.Drot(1, z.Off(zptr1+(*givcol)[1-1+(i-1)*2]-1-1), 1, z.Off(zptr1+(*givcol)[2-1+(i-1)*2]-1-1), 1, givnum.Get(0, i-1), givnum.Get(1, i-1))
		}
		for i = (*givptr)[curr+1-1]; i <= (*givptr)[curr+2-1]-1; i++ {
			goblas.Drot(1, z.Off(mid-1+(*givcol)[1-1+(i-1)*2]-1), 1, z.Off(mid-1+(*givcol)[2-1+(i-1)*2]-1), 1, givnum.Get(0, i-1), givnum.Get(1, i-1))
		}
		psiz1 = (*prmptr)[curr+1-1] - (*prmptr)[curr-1]
		psiz2 = (*prmptr)[curr+2-1] - (*prmptr)[curr+1-1]
		for i = 0; i <= psiz1-1; i++ {
			ztemp.Set(i+1-1, z.Get(zptr1+(*perm)[(*prmptr)[curr-1]+i-1]-1-1))
		}
		for i = 0; i <= psiz2-1; i++ {
			ztemp.Set(psiz1+i+1-1, z.Get(mid+(*perm)[(*prmptr)[curr+1-1]+i-1]-1-1))
		}

		//        Multiply Blocks at CURR and CURR+1
		//
		//        Determine size of these matrices.  We add HALF to the value of
		//        the SQRT in case the machine underestimates one of these
		//        square roots.
		bsiz1 = int(half + math.Sqrt(float64((*qptr)[curr+1-1]-(*qptr)[curr-1])))
		bsiz2 = int(half + math.Sqrt(float64((*qptr)[curr+2-1]-(*qptr)[curr+1-1])))
		if bsiz1 > 0 {
			err = goblas.Dgemv(Trans, bsiz1, bsiz1, one, q.MatrixOff((*qptr)[curr-1]-1, bsiz1, opts), bsiz1, ztemp, 1, zero, z.Off(zptr1-1), 1)
		}
		goblas.Dcopy(psiz1-bsiz1, ztemp.Off(bsiz1+1-1), 1, z.Off(zptr1+bsiz1-1), 1)
		if bsiz2 > 0 {
			err = goblas.Dgemv(Trans, bsiz2, bsiz2, one, q.MatrixOff((*qptr)[curr+1-1]-1, bsiz2, opts), bsiz2, ztemp.Off(psiz1+1-1), 1, zero, z.Off(mid-1), 1)
		}
		goblas.Dcopy(psiz2-bsiz2, ztemp.Off(psiz1+bsiz2+1-1), 1, z.Off(mid+bsiz2-1), 1)

		ptr = ptr + int(math.Pow(2, float64((*tlvls)-k)))
	}
}
