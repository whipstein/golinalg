package golapack

import (
	"fmt"
	"math"

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
func Zstedc(compz byte, n int, d, e *mat.Vector, z *mat.CMatrix, work *mat.CVector, lwork int, rwork *mat.Vector, lrwork int, iwork *[]int, liwork int) (info int, err error) {
	var lquery bool
	var eps, one, orgnrm, p, tiny, two, zero float64
	var finish, i, icompz, ii, j, k, lgn, liwmin, ll, lrwmin, lwmin, m, smlsiz, start int

	zero = 0.0
	one = 1.0
	two = 2.0

	//     Test the input parameters.
	lquery = (lwork == -1 || lrwork == -1 || liwork == -1)

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
		err = fmt.Errorf("icompz < 0: compz='%c'", compz)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if (z.Rows < 1) || (icompz > 0 && z.Rows < max(1, n)) {
		err = fmt.Errorf("(z.Rows < 1) || (icompz > 0 && z.Rows < max(1, n)): compz='%c', z.Rows=%v, n=%v", compz, z.Rows, n)
	}

	if err == nil {
		//        Compute the workspace requirements
		smlsiz = Ilaenv(9, "Zstedc", []byte{' '}, 0, 0, 0, 0)
		if n <= 1 || icompz == 0 {
			lwmin = 1
			liwmin = 1
			lrwmin = 1
		} else if n <= smlsiz {
			lwmin = 1
			liwmin = 1
			lrwmin = 2 * (n - 1)
		} else if icompz == 1 {
			lgn = int(math.Log(float64(n)) / math.Log(two))
			if pow(2, lgn) < n {
				lgn = lgn + 1
			}
			if pow(2, lgn) < n {
				lgn = lgn + 1
			}
			lwmin = n * n
			lrwmin = 1 + 3*n + 2*n*lgn + 4*pow(n, 2)
			liwmin = 6 + 6*n + 5*n*lgn
		} else if icompz == 2 {
			lwmin = 1
			lrwmin = 1 + 4*n + 2*pow(n, 2)
			liwmin = 3 + 5*n
		}
		work.SetRe(0, float64(lwmin))
		rwork.Set(0, float64(lrwmin))
		(*iwork)[0] = liwmin

		if lwork < lwmin && !lquery {
			err = fmt.Errorf("lwork < lwmin && !lquery: lwork=%v, lwmin=%v, lquery=%v", lwork, lwmin, lquery)
		} else if lrwork < lrwmin && !lquery {
			err = fmt.Errorf("lrwork < lrwmin && !lquery: lrwork=%v, lrwmin=%v, lquery=%v", lrwork, lrwmin, lquery)
		} else if liwork < liwmin && !lquery {
			err = fmt.Errorf("liwork < liwmin && !lquery: liwork=%v, liwmin=%v, lquery=%v", liwork, liwmin, lquery)
		}
	}

	if err != nil {
		gltest.Xerbla2("Zstedc", err)
		return
	} else if lquery {
		return
	}

	//     Quick return if possible
	if n == 0 {
		return
	}
	if n == 1 {
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
		if info, err = Dsterf(n, d, e); err != nil {
			panic(err)
		}
		goto label70
	}

	//     If N is smaller than the minimum divide size (SMLSIZ+1), then
	//     solve the problem with another solver.
	if n <= smlsiz {

		if info, err = Zsteqr(compz, n, d, e, z, rwork); err != nil {
			panic(err)
		}

	} else {
		//        If COMPZ = 'I', we simply call DSTEDC instead.
		if icompz == 2 {
			Dlaset(Full, n, n, zero, one, rwork.Matrix(n, opts))
			ll = n*n + 1
			if info, err = Dstedc('I', n, d, e, rwork.Matrix(n, opts), rwork.Off(ll-1), lrwork-ll+1, iwork, liwork); err != nil {
				panic(err)
			}
			for j = 1; j <= n; j++ {
				for i = 1; i <= n; i++ {
					z.Set(i-1, j-1, rwork.GetCmplx((j-1)*n+i-1))
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
		if start <= n {
			//           Let FINISH be the position of the next subdiagonal entry
			//           such that E( FINISH ) <= TINY or FINISH = N if no such
			//           subdiagonal exists.  The matrix identified by the elements
			//           between START and FINISH constitutes an independent
			//           sub-problem.
			finish = start
		label40:
			;
			if finish < n {
				tiny = eps * math.Sqrt(d.GetMag(finish-1)) * math.Sqrt(d.GetMag(finish))
				if e.GetMag(finish-1) > tiny {
					finish = finish + 1
					goto label40
				}
			}

			//           (Sub) Problem determined.  Compute its size and solve it.
			m = finish - start + 1
			if m > smlsiz {
				//              Scale.
				orgnrm = Dlanst('M', m, d.Off(start-1), e.Off(start-1))
				if err = Dlascl('G', 0, 0, orgnrm, one, m, 1, d.Off(start-1).Matrix(m, opts)); err != nil {
					panic(err)
				}
				if err = Dlascl('G', 0, 0, orgnrm, one, m-1, 1, e.Off(start-1).Matrix(m-1, opts)); err != nil {
					panic(err)
				}

				if info, err = Zlaed0(n, m, d.Off(start-1), e.Off(start-1), z.Off(0, start-1), work.CMatrix(n, opts), rwork, iwork); err != nil {
					panic(err)
				}
				if info > 0 {
					info = (info/(m+1)+start-1)*(n+1) + (info % (m + 1)) + start - 1
					goto label70
				}

				//              Scale back.
				if err = Dlascl('G', 0, 0, one, orgnrm, m, 1, d.Off(start-1).Matrix(m, opts)); err != nil {
					panic(err)
				}

			} else {
				if info, err = Dsteqr('I', m, d.Off(start-1), e.Off(start-1), rwork.Matrix(m, opts), rwork.Off(m*m)); err != nil {
					panic(err)
				}
				Zlacrm(n, m, z.Off(0, start-1), rwork.Matrix(m, opts), work.CMatrix(n, opts), rwork.Off(m*m))
				Zlacpy(Full, n, m, work.CMatrix(n, opts), z.Off(0, start-1))
				if info > 0 {
					info = start*(n+1) + finish
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
		for ii = 2; ii <= n; ii++ {
			i = ii - 1
			k = i
			p = d.Get(i - 1)
			for j = ii; j <= n; j++ {
				if d.Get(j-1) < p {
					k = j
					p = d.Get(j - 1)
				}
			}
			if k != i {
				d.Set(k-1, d.Get(i-1))
				d.Set(i-1, p)
				z.Off(0, k-1).CVector().Swap(n, z.Off(0, i-1).CVector(), 1, 1)
			}
		}
	}

label70:
	;
	work.SetRe(0, float64(lwmin))
	rwork.Set(0, float64(lrwmin))
	(*iwork)[0] = liwmin

	return
}
