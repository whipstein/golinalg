package golapack

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zlaed0 Using the divide and conquer method, ZLAED0 computes all eigenvalues
// of a symmetric tridiagonal matrix which is one diagonal block of
// those from reducing a dense or band Hermitian matrix and
// corresponding eigenvectors of the dense or band matrix.
func Zlaed0(qsiz, n *int, d, e *mat.Vector, q *mat.CMatrix, ldq *int, qstore *mat.CMatrix, ldqs *int, rwork *mat.Vector, iwork *[]int, info *int) {
	var temp, two float64
	var curlvl, curprb, curr, i, igivcl, igivnm, igivpt, indxq, iperm, iprmpt, iq, iqptr, iwrem, j, k, lgn, ll, matsiz, msd2, smlsiz, smm1, spm1, spm2, submat, subpbs, tlvls int

	two = 2.

	//     Test the input parameters.
	(*info) = 0

	//     IF( ICOMPQ .LT. 0 .OR. ICOMPQ .GT. 2 ) THEN
	//        INFO = -1
	//     ELSE IF( ( ICOMPQ .EQ. 1 ) .AND. ( QSIZ .LT. max( 0, N ) ) )
	//    $        THEN
	if (*qsiz) < max(0, *n) {
		(*info) = -1
	} else if (*n) < 0 {
		(*info) = -2
	} else if (*ldq) < max(1, *n) {
		(*info) = -6
	} else if (*ldqs) < max(1, *n) {
		(*info) = -8
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("ZLAED0"), -(*info))
		return
	}

	//     Quick return if possible
	if (*n) == 0 {
		return
	}

	smlsiz = Ilaenv(func() *int { y := 9; return &y }(), []byte("ZLAED0"), []byte{' '}, func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }())

	//     Determine the size and placement of the submatrices, and save in
	//     the leading elements of IWORK.
	(*iwork)[0] = (*n)
	subpbs = 1
	tlvls = 0
label10:
	;
	if (*iwork)[subpbs-1] > smlsiz {
		for j = subpbs; j >= 1; j-- {
			(*iwork)[2*j-1] = (*iwork)[j-1] + 1/2
			(*iwork)[2*j-1-1] = (*iwork)[j-1/2]
		}
		tlvls = tlvls + 1
		subpbs = 2 * subpbs
		goto label10
	}
	for j = 2; j <= subpbs; j++ {
		(*iwork)[j-1] = (*iwork)[j-1] + (*iwork)[j-1-1]
	}

	//     Divide the matrix into SUBPBS submatrices of size at most SMLSIZ+1
	//     using rank-1 modifications (cuts).
	spm1 = subpbs - 1
	for i = 1; i <= spm1; i++ {
		submat = (*iwork)[i-1] + 1
		smm1 = submat - 1
		d.Set(smm1-1, d.Get(smm1-1)-e.GetMag(smm1-1))
		d.Set(submat-1, d.Get(submat-1)-e.GetMag(smm1-1))
	}

	indxq = 4*(*n) + 3

	//     Set up workspaces for eigenvalues only/accumulate new vectors
	//     routine
	temp = math.Log(float64(*n)) / math.Log(two)
	lgn = int(temp)
	if pow(2, lgn) < (*n) {
		lgn = lgn + 1
	}
	if pow(2, lgn) < (*n) {
		lgn = lgn + 1
	}
	iprmpt = indxq + (*n) + 1
	iperm = iprmpt + (*n)*lgn
	iqptr = iperm + (*n)*lgn
	igivpt = iqptr + (*n) + 2
	igivcl = igivpt + (*n)*lgn

	igivnm = 1
	iq = igivnm + 2*(*n)*lgn
	iwrem = iq + pow(*n, 2) + 1
	//     Initialize pointers
	for i = 0; i <= subpbs; i++ {
		(*iwork)[iprmpt+i-1] = 1
		(*iwork)[igivpt+i-1] = 1
	}
	(*iwork)[iqptr-1] = 1

	//     Solve each submatrix eigenproblem at the bottom of the divide and
	//     conquer tree.
	curr = 0
	for i = 0; i <= spm1; i++ {
		if i == 0 {
			submat = 1
			matsiz = (*iwork)[0]
		} else {
			submat = (*iwork)[i-1] + 1
			matsiz = (*iwork)[i] - (*iwork)[i-1]
		}
		ll = iq - 1 + (*iwork)[iqptr+curr-1]
		Dsteqr('I', &matsiz, d.Off(submat-1), e.Off(submat-1), rwork.MatrixOff(ll-1, matsiz, opts), &matsiz, rwork, info)
		Zlacrm(qsiz, &matsiz, q.Off(0, submat-1), ldq, rwork.MatrixOff(ll-1, matsiz, opts), &matsiz, qstore.Off(0, submat-1), ldqs, rwork.Off(iwrem-1))
		(*iwork)[iqptr+curr] = (*iwork)[iqptr+curr-1] + pow(matsiz, 2)
		curr = curr + 1
		if (*info) > 0 {
			(*info) = submat*((*n)+1) + submat + matsiz - 1
			return
		}
		k = 1
		for j = submat; j <= (*iwork)[i]; j++ {
			(*iwork)[indxq+j-1] = k
			k = k + 1
		}
	}

	//     Successively merge eigensystems of adjacent submatrices
	//     into eigensystem for the corresponding larger matrix.
	//
	//     while ( SUBPBS > 1 )
	curlvl = 1
label80:
	;
	if subpbs > 1 {
		spm2 = subpbs - 2
		for i = 0; i <= spm2; i += 2 {
			if i == 0 {
				submat = 1
				matsiz = (*iwork)[1]
				msd2 = (*iwork)[0]
				curprb = 0
			} else {
				submat = (*iwork)[i-1] + 1
				matsiz = (*iwork)[i+2-1] - (*iwork)[i-1]
				msd2 = matsiz / 2
				curprb = curprb + 1
			}

			//     Merge lower order eigensystems (of size MSD2 and MATSIZ - MSD2)
			//     into an eigensystem of size MATSIZ.  ZLAED7 handles the case
			//     when the eigenvectors of a full or band Hermitian matrix (which
			//     was reduced to tridiagonal form) are desired.
			//
			//     I am free to use Q as a valuable working space until Loop 150.
			Zlaed7(&matsiz, &msd2, qsiz, &tlvls, &curlvl, &curprb, d.Off(submat-1), qstore.Off(0, submat-1), ldqs, e.GetPtr(submat+msd2-1-1), toSlice(iwork, indxq+submat-1), rwork.Off(iq-1), toSlice(iwork, iqptr-1), toSlice(iwork, iprmpt-1), toSlice(iwork, iperm-1), toSlice(iwork, igivpt-1), toSlice(iwork, igivcl-1), rwork.MatrixOff(igivnm-1, 2, opts), q.CVector(0, submat-1), rwork.Off(iwrem-1), toSlice(iwork, subpbs), info)
			if (*info) > 0 {
				(*info) = submat*((*n)+1) + submat + matsiz - 1
				return
			}
			(*iwork)[i/2] = (*iwork)[i+2-1]
		}
		subpbs = subpbs / 2
		curlvl = curlvl + 1
		goto label80
	}

	//     end while
	//
	//     Re-merge the eigenvalues/vectors which were deflated at the final
	//     merge step.
	for i = 1; i <= (*n); i++ {
		j = (*iwork)[indxq+i-1]
		rwork.Set(i-1, d.Get(j-1))
		goblas.Zcopy(*qsiz, qstore.CVector(0, j-1, 1), q.CVector(0, i-1, 1))
	}
	goblas.Dcopy(*n, rwork.Off(0, 1), d.Off(0, 1))
}
