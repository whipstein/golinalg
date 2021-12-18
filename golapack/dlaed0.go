package golapack

import (
	"fmt"
	"math"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dlaed0 computes all eigenvalues and corresponding eigenvectors of a
// symmetric tridiagonal matrix using the divide and conquer method.
func Dlaed0(icompq, qsiz, n int, d, e *mat.Vector, q, qstore *mat.Matrix, work *mat.Vector, iwork *[]int) (info int, err error) {
	var one, temp, two, zero float64
	var curlvl, curprb, curr, i, igivcl, igivnm, igivpt, indxq, iperm, iprmpt, iq, iqptr, iwrem, j, k, lgn, matsiz, msd2, smlsiz, smm1, spm1, spm2, submat, subpbs, tlvls int

	zero = 0.
	one = 1.
	two = 2.

	//     Test the input parameters.
	if icompq < 0 || icompq > 2 {
		err = fmt.Errorf("icompq < 0 || icompq > 2: icompq=%v", icompq)
	} else if (icompq == 1) && (qsiz < max(0, n)) {
		err = fmt.Errorf("(icompq == 1) && (qsiz < max(0, n)): icompq=%v, qsiz=%v", icompq, qsiz)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if q.Rows < max(1, n) {
		err = fmt.Errorf("q.Rows < max(1, n): q.Rows=%v, n=%v", q.Rows, n)
	} else if qstore.Rows < max(1, n) {
		err = fmt.Errorf("qstore.Rows < max(1, n): qstore.Rows=%v, n=%v", qstore.Rows, n)
	}
	if err != nil {
		gltest.Xerbla2("Dlaed0", err)
		return
	}

	//     Quick return if possible
	if n == 0 {
		return
	}

	smlsiz = Ilaenv(9, "Dlaed0", []byte{' '}, 0, 0, 0, 0)

	//     Determine the size and placement of the submatrices, and save in
	//     the leading elements of IWORK.
	(*iwork)[0] = n
	subpbs = 1
	tlvls = 0
label10:
	;
	if (*iwork)[subpbs-1] > smlsiz {
		for j = subpbs; j >= 1; j-- {
			(*iwork)[2*j-1] = ((*iwork)[j-1] + 1) / 2
			(*iwork)[2*j-1-1] = (*iwork)[j-1] / 2
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
		d.Set(smm1-1, d.Get(smm1-1)-math.Abs(e.Get(smm1-1)))
		d.Set(submat-1, d.Get(submat-1)-math.Abs(e.Get(smm1-1)))
	}

	indxq = 4*n + 3
	if icompq != 2 {
		//        Set up workspaces for eigenvalues only/accumulate new vectors
		//        routine
		temp = math.Log(float64(n)) / math.Log(two)
		lgn = int(temp)
		if int(math.Pow(2, float64(lgn))) < n {
			lgn = lgn + 1
		}
		if int(math.Pow(2, float64(lgn))) < n {
			lgn = lgn + 1
		}
		iprmpt = indxq + n + 1
		iperm = iprmpt + n*lgn
		iqptr = iperm + n*lgn
		igivpt = iqptr + n + 2
		igivcl = igivpt + n*lgn

		igivnm = 1
		iq = igivnm + 2*n*lgn
		iwrem = iq + pow(n, 2) + 1

		//        Initialize pointers
		for i = 0; i <= subpbs; i++ {
			(*iwork)[iprmpt+i-1] = 1
			(*iwork)[igivpt+i-1] = 1
		}
		(*iwork)[iqptr-1] = 1
	}

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
		if icompq == 2 {
			if info, err = Dsteqr('I', matsiz, d.Off(submat-1), e.Off(submat-1), q.Off(submat-1, submat-1), work); err != nil {
				panic(err)
			}
			if info != 0 {
				goto label130
			}
		} else {
			if info, err = Dsteqr('I', matsiz, d.Off(submat-1), e.Off(submat-1), work.Off(iq-1+(*iwork)[iqptr+curr-1]-1).Matrix(matsiz, opts), work); err != nil {
				panic(err)
			}
			if info != 0 {
				goto label130
			}
			if icompq == 1 {
				if err = qstore.Off(0, submat-1).Gemm(NoTrans, NoTrans, qsiz, matsiz, matsiz, one, q.Off(0, submat-1), work.Off(iq-1+(*iwork)[iqptr+curr-1]-1).Matrix(matsiz, opts), zero); err != nil {
					panic(err)
				}
			}
			(*iwork)[iqptr+curr] = (*iwork)[iqptr+curr-1] + int(math.Pow(float64(matsiz), 2))
			curr = curr + 1
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
			//     into an eigensystem of size MATSIZ.
			//     DLAED1 is used only for the full eigensystem of a tridiagonal
			//     matrix.
			//     DLAED7 handles the cases in which eigenvalues only or eigenvalues
			//     and eigenvectors of a full symmetric matrix (which was reduced to
			//     tridiagonal form) are desired.
			if icompq == 2 {
				if info, err = Dlaed1(matsiz, d.Off(submat-1), q.Off(submat-1, submat-1), toSlice(iwork, indxq+submat-1), e.Get(submat+msd2-1-1), msd2, work, toSlice(iwork, subpbs)); err != nil {
					panic(err)
				}
			} else {
				if info, err = Dlaed7(icompq, matsiz, qsiz, tlvls, curlvl, curprb, d.Off(submat-1), qstore.Off(0, submat-1), toSlice(iwork, indxq+submat-1), e.Get(submat+msd2-1-1), msd2, work.Off(iq-1), toSlice(iwork, iqptr-1), toSlice(iwork, iprmpt-1), toSlice(iwork, iperm-1), toSlice(iwork, igivpt-1), toSlice(iwork, igivcl-1), work.Off(igivnm-1).Matrix(2, opts), work.Off(iwrem-1), toSlice(iwork, subpbs)); err != nil {
					panic(err)
				}
			}
			if info != 0 {
				goto label130
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
	if icompq == 1 {
		for i = 1; i <= n; i++ {
			j = (*iwork)[indxq+i-1]
			work.Set(i-1, d.Get(j-1))
			q.Off(0, i-1).Vector().Copy(qsiz, qstore.Off(0, j-1).Vector(), 1, 1)
		}
		d.Copy(n, work, 1, 1)
	} else if icompq == 2 {
		for i = 1; i <= n; i++ {
			j = (*iwork)[indxq+i-1]
			work.Set(i-1, d.Get(j-1))
			work.Off(n*i).Copy(n, q.Off(0, j-1).Vector(), 1, 1)
		}
		d.Copy(n, work, 1, 1)
		Dlacpy(Full, n, n, work.Off(n).Matrix(n, opts), q)
	} else {
		for i = 1; i <= n; i++ {
			j = (*iwork)[indxq+i-1]
			work.Set(i-1, d.Get(j-1))
		}
		d.Copy(n, work, 1, 1)
	}
	return

label130:
	;
	info = submat*(n+1) + submat + matsiz - 1

	return
}
