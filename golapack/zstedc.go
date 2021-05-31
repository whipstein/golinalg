package golapack

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zstedc computes all eigenvalues and, optionally, eigenvectors of a
// symmetric tridiagonal matrix using the divide and conquer method.
// The eigenvectors of a full or band complex Hermitian matrix can also
// be found if ZHETRD or ZHPTRD or ZHBTRD has been used to reduce this
// matrix to tridiagonal form.
//
// This code makes very mild assumptions about floating point
// arithmetic. It will work on machines with a guard digit in
// add/subtract, or on those binary machines without guard digits
// which subtract like the Cray X-MP, Cray Y-MP, Cray C-90, or Cray-2.
// It could conceivably fail on hexadecimal or decimal machines
// without guard digits, but we know of none.  See DLAED3 for details.
func Zstedc(compz byte, n *int, d, e *mat.Vector, z *mat.CMatrix, ldz *int, work *mat.CVector, lwork *int, rwork *mat.Vector, lrwork *int, iwork *[]int, liwork, info *int) {
	var lquery bool
	var eps, one, orgnrm, p, tiny, two, zero float64
	var finish, i, icompz, ii, j, k, lgn, liwmin, ll, lrwmin, lwmin, m, smlsiz, start int

	zero = 0.0
	one = 1.0
	two = 2.0

	//     Test the input parameters.
	(*info) = 0
	lquery = ((*lwork) == -1 || (*lrwork) == -1 || (*liwork) == -1)

	if compz == 'N' {
		icompz = 0
	} else if compz == 'V' {
		icompz = 1
	} else if compz == 'I' {
		icompz = 2
	} else {
		icompz = -1
	}
	if icompz < 0 {
		(*info) = -1
	} else if (*n) < 0 {
		(*info) = -2
	} else if ((*ldz) < 1) || (icompz > 0 && (*ldz) < maxint(1, *n)) {
		(*info) = -6
	}

	if (*info) == 0 {
		//        Compute the workspace requirements
		smlsiz = Ilaenv(func() *int { y := 9; return &y }(), []byte("ZSTEDC"), []byte{' '}, func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }())
		if (*n) <= 1 || icompz == 0 {
			lwmin = 1
			liwmin = 1
			lrwmin = 1
		} else if (*n) <= smlsiz {
			lwmin = 1
			liwmin = 1
			lrwmin = 2 * ((*n) - 1)
		} else if icompz == 1 {
			lgn = int(math.Log(float64(*n)) / math.Log(two))
			if powint(2, lgn) < (*n) {
				lgn = lgn + 1
			}
			if powint(2, lgn) < (*n) {
				lgn = lgn + 1
			}
			lwmin = (*n) * (*n)
			lrwmin = 1 + 3*(*n) + 2*(*n)*lgn + 4*powint(*n, 2)
			liwmin = 6 + 6*(*n) + 5*(*n)*lgn
		} else if icompz == 2 {
			lwmin = 1
			lrwmin = 1 + 4*(*n) + 2*powint(*n, 2)
			liwmin = 3 + 5*(*n)
		}
		work.SetRe(0, float64(lwmin))
		rwork.Set(0, float64(lrwmin))
		(*iwork)[0] = liwmin

		if (*lwork) < lwmin && !lquery {
			(*info) = -8
		} else if (*lrwork) < lrwmin && !lquery {
			(*info) = -10
		} else if (*liwork) < liwmin && !lquery {
			(*info) = -12
		}
	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("ZSTEDC"), -(*info))
		return
	} else if lquery {
		return
	}

	//     Quick return if possible
	if (*n) == 0 {
		return
	}
	if (*n) == 1 {
		if icompz != 0 {
			z.SetRe(0, 0, one)
		}
		return
	}

	//     If the following conditional clause is removed, then the routine
	//     will use the Divide and Conquer routine to compute only the
	//     eigenvalues, which requires (3N + 3N**2) real workspace and
	//     (2 + 5N + 2N lg(N)) integer workspace.
	//     Since on many architectures DSTERF is much faster than any other
	//     algorithm for finding eigenvalues only, it is used here
	//     as the default. If the conditional clause is removed, then
	//     information on the size of workspace needs to be changed.
	//
	//     If COMPZ = 'N', use DSTERF to compute the eigenvalues.
	if icompz == 0 {
		Dsterf(n, d, e, info)
		goto label70
	}

	//     If N is smaller than the minimum divide size (SMLSIZ+1), then
	//     solve the problem with another solver.
	if (*n) <= smlsiz {

		Zsteqr(compz, n, d, e, z, ldz, rwork, info)

	} else {
		//        If COMPZ = 'I', we simply call DSTEDC instead.
		if icompz == 2 {
			Dlaset('F', n, n, &zero, &one, rwork.Matrix(*n, opts), n)
			ll = (*n)*(*n) + 1
			Dstedc('I', n, d, e, rwork.Matrix(*n, opts), n, rwork.Off(ll-1), toPtr((*lrwork)-ll+1), iwork, liwork, info)
			for j = 1; j <= (*n); j++ {
				for i = 1; i <= (*n); i++ {
					z.Set(i-1, j-1, rwork.GetCmplx((j-1)*(*n)+i-1))
				}
			}
			goto label70
		}

		//        From now on, only option left to be handled is COMPZ = 'V',
		//        i.e. ICOMPZ = 1.
		//
		//        Scale.
		orgnrm = Dlanst('M', n, d, e)
		if orgnrm == zero {
			goto label70
		}

		eps = Dlamch(Epsilon)

		start = 1

		//        while ( START <= N )
	label30:
		;
		if start <= (*n) {
			//           Let FINISH be the position of the next subdiagonal entry
			//           such that E( FINISH ) <= TINY or FINISH = N if no such
			//           subdiagonal exists.  The matrix identified by the elements
			//           between START and FINISH constitutes an independent
			//           sub-problem.
			finish = start
		label40:
			;
			if finish < (*n) {
				tiny = eps * math.Sqrt(d.GetMag(finish-1)) * math.Sqrt(d.GetMag(finish+1-1))
				if e.GetMag(finish-1) > tiny {
					finish = finish + 1
					goto label40
				}
			}

			//           (Sub) Problem determined.  Compute its size and solve it.
			m = finish - start + 1
			if m > smlsiz {
				//              Scale.
				orgnrm = Dlanst('M', &m, d.Off(start-1), e.Off(start-1))
				Dlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &orgnrm, &one, &m, func() *int { y := 1; return &y }(), d.MatrixOff(start-1, m, opts), &m, info)
				Dlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &orgnrm, &one, toPtr(m-1), func() *int { y := 1; return &y }(), e.MatrixOff(start-1, m-1, opts), toPtr(m-1), info)

				Zlaed0(n, &m, d.Off(start-1), e.Off(start-1), z.Off(0, start-1), ldz, work.CMatrix(*n, opts), n, rwork, iwork, info)
				if (*info) > 0 {
					(*info) = ((*info)/(m+1)+start-1)*((*n)+1) + (*info % (m + 1)) + start - 1
					goto label70
				}

				//              Scale back.
				Dlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &one, &orgnrm, &m, func() *int { y := 1; return &y }(), d.MatrixOff(start-1, m, opts), &m, info)

			} else {
				Dsteqr('I', &m, d.Off(start-1), e.Off(start-1), rwork.Matrix(m, opts), &m, rwork.Off(m*m+1-1), info)
				Zlacrm(n, &m, z.Off(0, start-1), ldz, rwork.Matrix(m, opts), &m, work.CMatrix(*n, opts), n, rwork.Off(m*m+1-1))
				Zlacpy('A', n, &m, work.CMatrix(*n, opts), n, z.Off(0, start-1), ldz)
				if (*info) > 0 {
					(*info) = start*((*n)+1) + finish
					goto label70
				}
			}

			start = finish + 1
			goto label30
		}

		//        endwhile
		//
		//
		//        Use Selection Sort to minimize swaps of eigenvectors
		for ii = 2; ii <= (*n); ii++ {
			i = ii - 1
			k = i
			p = d.Get(i - 1)
			for j = ii; j <= (*n); j++ {
				if d.Get(j-1) < p {
					k = j
					p = d.Get(j - 1)
				}
			}
			if k != i {
				d.Set(k-1, d.Get(i-1))
				d.Set(i-1, p)
				goblas.Zswap(n, z.CVector(0, i-1), func() *int { y := 1; return &y }(), z.CVector(0, k-1), func() *int { y := 1; return &y }())
			}
		}
	}

label70:
	;
	work.SetRe(0, float64(lwmin))
	rwork.Set(0, float64(lrwmin))
	(*iwork)[0] = liwmin
}
