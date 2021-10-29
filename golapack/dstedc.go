package golapack

import (
	"fmt"
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dstedc computes all eigenvalues and, optionally, eigenvectors of a
// symmetric tridiagonal matrix using the divide and conquer method.
// The eigenvectors of a full or band real symmetric matrix can also be
// found if DSYTRD or DSPTRD or DSBTRD has been used to reduce this
// matrix to tridiagonal form.
//
// This code makes very mild assumptions about floating point
// arithmetic. It will work on machines with a guard digit in
// add/subtract, or on those binary machines without guard digits
// which subtract like the Cray X-MP, Cray Y-MP, Cray C-90, or Cray-2.
// It could conceivably fail on hexadecimal or decimal machines
// without guard digits, but we know of none.  See DLAED3 for details.
func Dstedc(compz byte, n int, d, e *mat.Vector, z *mat.Matrix, work *mat.Vector, lwork int, iwork *[]int, liwork int) (info int, err error) {
	var lquery bool
	var eps, one, orgnrm, p, tiny, two, zero float64
	var finish, i, icompz, ii, j, k, lgn, liwmin, lwmin, m, smlsiz, start, storez, strtrw int

	zero = 0.0
	one = 1.0
	two = 2.0

	//     Test the input parameters.
	lquery = (lwork == -1 || liwork == -1)

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
		smlsiz = Ilaenv(9, "Dstedc", []byte{' '}, 0, 0, 0, 0)
		if n <= 1 || icompz == 0 {
			liwmin = 1
			lwmin = 1
		} else if n <= smlsiz {
			liwmin = 1
			lwmin = 2 * (n - 1)
		} else {
			lgn = int(math.Log(float64(n)) / math.Log(two))
			if int(math.Pow(2, float64(lgn))) < n {
				lgn = lgn + 1
			}
			if int(math.Pow(2, float64(lgn))) < n {
				lgn = lgn + 1
			}
			if icompz == 1 {
				lwmin = 1 + 3*n + 2*n*lgn + 4*pow(n, 2)
				liwmin = 6 + 6*n + 5*n*lgn
			} else if icompz == 2 {
				lwmin = 1 + 4*n + pow(n, 2)
				liwmin = 3 + 5*n
			}
		}
		work.Set(0, float64(lwmin))
		(*iwork)[0] = liwmin

		if lwork < lwmin && !lquery {
			err = fmt.Errorf("lwork < lwmin && !lquery: lwork=%v, lwmin=%v, lquery=%v", lwork, lwmin, lquery)
		} else if liwork < liwmin && !lquery {
			err = fmt.Errorf("liwork < liwmin && !lquery: liwork=%v, liwmin=%v, lquery=%v", liwork, liwmin, lquery)
		}
	}

	if err != nil {
		gltest.Xerbla2("Dstedc", err)
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
			z.Set(0, 0, one)
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
		goto label50
	}

	//     If N is smaller than the minimum divide size (SMLSIZ+1), then
	//     solve the problem with another solver.
	if n <= smlsiz {

		if info, err = Dsteqr(compz, n, d, e, z, work); err != nil {
			panic(err)
		}

	} else {
		//        If COMPZ = 'V', the Z matrix must be stored elsewhere for later
		//        use.
		if icompz == 1 {
			storez = 1 + n*n
		} else {
			storez = 1
		}

		if icompz == 2 {
			Dlaset(Full, n, n, zero, one, z)
		}

		//        Scale.
		orgnrm = Dlanst('M', n, d, e)
		if orgnrm == zero {
			goto label50
		}

		eps = Dlamch(Epsilon)

		start = 1

		//        while ( START <= N )
	label10:
		;
		if start <= n {
			//           Let FINISH be the position of the next subdiagonal entry
			//           such that E( FINISH ) <= TINY or FINISH = N if no such
			//           subdiagonal exists.  The matrix identified by the elements
			//           between START and FINISH constitutes an independent
			//           sub-problem.
			finish = start
		label20:
			;
			if finish < n {
				tiny = eps * math.Sqrt(math.Abs(d.Get(finish-1))) * math.Sqrt(math.Abs(d.Get(finish)))
				if math.Abs(e.Get(finish-1)) > tiny {
					finish = finish + 1
					goto label20
				}
			}

			//           (Sub) Problem determined.  Compute its size and solve it.
			m = finish - start + 1
			if m == 1 {
				start = finish + 1
				goto label10
			}
			if m > smlsiz {
				//              Scale.
				orgnrm = Dlanst('M', m, d.Off(start-1), e.Off(start-1))
				if err = Dlascl('G', 0, 0, orgnrm, one, m, 1, d.MatrixOff(start-1, m, opts)); err != nil {
					panic(err)
				}
				if err = Dlascl('G', 0, 0, orgnrm, one, m-1, 1, e.MatrixOff(start-1, m-1, opts)); err != nil {
					panic(err)
				}

				if icompz == 1 {
					strtrw = 1
				} else {
					strtrw = start
				}
				if info, err = Dlaed0(icompz, n, m, d.Off(start-1), e.Off(start-1), z.Off(strtrw-1, start-1), work.MatrixOff(0, n, opts), work.Off(storez-1), iwork); err != nil {
					panic(err)
				}
				if info != 0 {
					info = (info/(m+1)+start-1)*(n+1) + info%(m+1) + start - 1
					goto label50
				}

				//              Scale back.
				if err = Dlascl('G', 0, 0, one, orgnrm, m, 1, d.MatrixOff(start-1, m, opts)); err != nil {
					panic(err)
				}

			} else {
				if icompz == 1 {
					//                 Since QR won't update a Z matrix which is larger than
					//                 the length of D, we must solve the sub-problem in a
					//                 workspace and then multiply back into Z.
					if info, err = Dsteqr('I', m, d.Off(start-1), e.Off(start-1), work.Matrix(m, opts), work.Off(m*m)); err != nil {
						panic(err)
					}
					Dlacpy(Full, n, m, z.Off(0, start-1), work.MatrixOff(storez-1, n, opts))
					if err = goblas.Dgemm(NoTrans, NoTrans, n, m, m, one, work.MatrixOff(storez-1, n, opts), work.Matrix(m, opts), zero, z.Off(0, start-1)); err != nil {
						panic(err)
					}
				} else if icompz == 2 {
					if info, err = Dsteqr('I', m, d.Off(start-1), e.Off(start-1), z.Off(start-1, start-1), work); err != nil {
						panic(err)
					}
				} else {
					if info, err = Dsterf(m, d.Off(start-1), e.Off(start-1)); err != nil {
						panic(err)
					}
				}
				if info != 0 {
					info = start*(n+1) + finish
					goto label50
				}
			}

			start = finish + 1
			goto label10
		}

		//        endwhile
		if icompz == 0 {
			//          Use Quick Sort
			if err = Dlasrt('I', n, d); err != nil {
				panic(err)
			}

		} else {
			//          Use Selection Sort to minimize swaps of eigenvectors
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
					goblas.Dswap(n, z.Vector(0, i-1, 1), z.Vector(0, k-1, 1))
				}
			}
		}
	}

label50:
	;
	work.Set(0, float64(lwmin))
	(*iwork)[0] = liwmin

	return
}
