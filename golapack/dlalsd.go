package golapack

import (
	"fmt"
	"math"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dlalsd uses the singular value decomposition of A to solve the least
// squares problem of finding X to minimize the Euclidean norm of each
// column of A*X-B, where A is N-by-N upper bidiagonal, and X and B
// are N-by-NRHS. The solution X overwrites B.
//
// The singular values of A smaller than RCOND times the largest
// singular value are treated as zero in solving the least squares
// problem; in this case a minimum norm solution is returned.
// The actual singular values are returned in D in ascending order.
//
// This code makes very mild assumptions about floating point
// arithmetic. It will work on machines with a guard digit in
// add/subtract, or on those binary machines without guard digits
// which subtract like the Cray XMP, Cray YMP, Cray C 90, or Cray 2.
// It could conceivably fail on hexadecimal or decimal machines
// without guard digits, but we know of none.
func Dlalsd(uplo mat.MatUplo, smlsiz, n, nrhs int, d, e *mat.Vector, b *mat.Matrix, rcond float64, work *mat.Vector, iwork *[]int) (rank, info int, err error) {
	var cs, eps, one, orgnrm, r, rcnd, sn, tol, two, zero float64
	var bx, bxst, c, difl, difr, givcol, givnum, givptr, i, icmpq1, icmpq2, iwk, j, k, nlvl, nm1, nsize, nsub, nwork, perm, poles, s, sizei, smlszp, sqre, st, st1, u, vt, z int

	zero = 0.0
	one = 1.0
	two = 2.0

	//     Test the input parameters.
	if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if nrhs < 1 {
		err = fmt.Errorf("nrhs < 1: nrhs=%v", nrhs)
	} else if (b.Rows < 1) || (b.Rows < n) {
		err = fmt.Errorf("(b.Rows < 1) || (b.Rows < n): b.Rows=%v, n=%v", b.Rows, n)
	}
	if err != nil {
		gltest.Xerbla2("Dlalsd", err)
		return
	}

	eps = Dlamch(Epsilon)

	//     Set up the tolerance.
	if (rcond <= zero) || (rcond >= one) {
		rcnd = eps
	} else {
		rcnd = rcond
	}

	rank = 0

	//     Quick return if possible.
	if n == 0 {
		return
	} else if n == 1 {
		if d.Get(0) == zero {
			Dlaset(Full, 1, nrhs, zero, zero, b)
		} else {
			rank = 1
			if err = Dlascl('G', 0, 0, d.Get(0), one, 1, nrhs, b); err != nil {
				panic(err)
			}
			d.Set(0, math.Abs(d.Get(0)))
		}
		return
	}

	//     Rotate the matrix if it is lower bidiagonal.
	if uplo == 'L' {
		for i = 1; i <= n-1; i++ {
			cs, sn, r = Dlartg(d.Get(i-1), e.Get(i-1))
			d.Set(i-1, r)
			e.Set(i-1, sn*d.Get(i))
			d.Set(i, cs*d.Get(i))
			if nrhs == 1 {
				b.Off(i, 0).Vector().Rot(1, b.Off(i-1, 0).Vector(), 1, 1, cs, sn)
			} else {
				work.Set(i*2-1-1, cs)
				work.Set(i*2-1, sn)
			}
		}
		if nrhs > 1 {
			for i = 1; i <= nrhs; i++ {
				for j = 1; j <= n-1; j++ {
					cs = work.Get(j*2 - 1 - 1)
					sn = work.Get(j*2 - 1)
					b.Off(j, i-1).Vector().Rot(1, b.Off(j-1, i-1).Vector(), 1, 1, cs, sn)
				}
			}
		}
	}

	//     Scale.
	nm1 = n - 1
	orgnrm = Dlanst('M', n, d, e)
	if orgnrm == zero {
		Dlaset(Full, n, nrhs, zero, zero, b)
		return
	}

	if err = Dlascl('G', 0, 0, orgnrm, one, n, 1, d.Matrix(n, opts)); err != nil {
		panic(err)
	}
	if err = Dlascl('G', 0, 0, orgnrm, one, nm1, 1, e.Matrix(nm1, opts)); err != nil {
		panic(err)
	}

	//     If N is smaller than the minimum divide size SMLSIZ, then solve
	//     the problem with another solver.
	if n <= smlsiz {
		nwork = 1 + n*n
		Dlaset(Full, n, n, zero, one, work.Matrix(n, opts))
		if info, err = Dlasdq(Upper, 0, n, n, 0, nrhs, d, e, work.Matrix(n, opts), work.Matrix(n, opts), b, work.Off(nwork-1)); err != nil {
			panic(err)
		}
		if info != 0 {
			return
		}
		tol = rcnd * math.Abs(d.Get(d.Iamax(n, 1)-1))
		for i = 1; i <= n; i++ {
			if d.Get(i-1) <= tol {
				Dlaset(Full, 1, nrhs, zero, zero, b.Off(i-1, 0))
			} else {
				if err = Dlascl('G', 0, 0, d.Get(i-1), one, 1, nrhs, b.Off(i-1, 0)); err != nil {
					panic(err)
				}
				rank = rank + 1
			}
		}
		if err = work.Off(nwork-1).Matrix(n, opts).Gemm(Trans, NoTrans, n, nrhs, n, one, work.Matrix(n, opts), b, zero); err != nil {
			panic(err)
		}
		Dlacpy(Full, n, nrhs, work.Off(nwork-1).Matrix(n, opts), b)

		//        Unscale.
		if err = Dlascl('G', 0, 0, one, orgnrm, n, 1, d.Matrix(n, opts)); err != nil {
			panic(err)
		}
		if err = Dlasrt('D', n, d); err != nil {
			panic(err)
		}
		if err = Dlascl('G', 0, 0, orgnrm, one, n, nrhs, b); err != nil {
			panic(err)
		}

		return
	}

	//     Book-keeping and setting up some constants.
	nlvl = int(math.Log(float64(n)/float64(smlsiz+1))/math.Log(two)) + 1

	smlszp = smlsiz + 1

	u = 1
	vt = 1 + smlsiz*n
	difl = vt + smlszp*n
	difr = difl + nlvl*n
	z = difr + nlvl*n*2
	c = z + nlvl*n
	s = c + n
	poles = s + n
	givnum = poles + 2*nlvl*n
	bx = givnum + 2*nlvl*n
	nwork = bx + n*nrhs

	sizei = 1 + n
	k = sizei + n
	givptr = k + n
	perm = givptr + n
	givcol = perm + nlvl*n
	iwk = givcol + nlvl*n*2

	st = 1
	sqre = 0
	icmpq1 = 1
	icmpq2 = 0
	nsub = 0

	for i = 1; i <= n; i++ {
		if math.Abs(d.Get(i-1)) < eps {
			d.Set(i-1, math.Copysign(eps, d.Get(i-1)))
		}
	}

	for i = 1; i <= nm1; i++ {
		if (math.Abs(e.Get(i-1)) < eps) || (i == nm1) {
			nsub = nsub + 1
			(*iwork)[nsub-1] = st

			//           Subproblem found. First determine its size and then
			//           apply divide and conquer on it.
			if i < nm1 {
				//              A subproblem with E(I) small for I < NM1.
				nsize = i - st + 1
				(*iwork)[sizei+nsub-1-1] = nsize
			} else if math.Abs(e.Get(i-1)) >= eps {
				//              A subproblem with E(NM1) not too small but I = NM1.
				nsize = n - st + 1
				(*iwork)[sizei+nsub-1-1] = nsize
			} else {
				//              A subproblem with E(NM1) small. This implies an
				//              1-by-1 subproblem at D(N), which is not solved
				//              explicitly.
				nsize = i - st + 1
				(*iwork)[sizei+nsub-1-1] = nsize
				nsub = nsub + 1
				(*iwork)[nsub-1] = n
				(*iwork)[sizei+nsub-1-1] = 1
				work.Off(bx+nm1-1).Copy(nrhs, b.Off(n-1, 0).Vector(), b.Rows, n)
			}
			st1 = st - 1
			if nsize == 1 {
				//              This is a 1-by-1 subproblem and is not solved
				//              explicitly.
				work.Off(bx+st1-1).Copy(nrhs, b.Off(st-1, 0).Vector(), b.Rows, n)
			} else if nsize <= smlsiz {
				//              This is a small subproblem and is solved by DLASDQ.
				Dlaset(Full, nsize, nsize, zero, one, work.Off(vt+st1-1).Matrix(n, opts))
				if info, err = Dlasdq(Upper, 0, nsize, nsize, 0, nrhs, d.Off(st-1), e.Off(st-1), work.Off(vt+st1-1).Matrix(n, opts), work.Off(nwork-1).Matrix(n, opts), b.Off(st-1, 0), work.Off(nwork-1)); err != nil {
					panic(err)
				}
				if info != 0 {
					return
				}
				Dlacpy(Full, nsize, nrhs, b.Off(st-1, 0), work.Off(bx+st1-1).Matrix(n, opts))
			} else {
				//              A large problem. Solve it using divide and conquer.
				if info, err = Dlasda(icmpq1, smlsiz, nsize, sqre, d.Off(st-1), e.Off(st-1), work.Off(u+st1-1).Matrix(n, opts), work.Off(vt+st1-1).Matrix(n, opts), toSlice(iwork, k+st1-1), work.Off(difl+st1-1).Matrix(n, opts), work.Off(difr+st1-1).Matrix(n, opts), work.Off(z+st1-1).Matrix(n, opts), work.Off(poles+st1-1).Matrix(n, opts), toSlice(iwork, givptr+st1-1), toSlice(iwork, givcol+st1-1), n, toSlice(iwork, perm+st1-1), work.Off(givnum+st1-1).Matrix(n, opts), work.Off(c+st1-1), work.Off(s+st1-1), work.Off(nwork-1), toSlice(iwork, iwk-1)); err != nil {
					panic(err)
				}
				if info != 0 {
					return
				}
				bxst = bx + st1
				if err = Dlalsa(icmpq2, smlsiz, nsize, nrhs, b.Off(st-1, 0), work.Off(bxst-1).Matrix(n, opts), work.Off(u+st1-1).Matrix(n, opts), work.Off(vt+st1-1).Matrix(n, opts), toSlice(iwork, k+st1-1), work.Off(difl+st1-1).Matrix(n, opts), work.Off(difr+st1-1).Matrix(n, opts), work.Off(z+st1-1).Matrix(n, opts), work.Off(poles+st1-1).Matrix(n, opts), toSlice(iwork, givptr+st1-1), toSlice(iwork, givcol+st1-1), n, toSlice(iwork, perm+st1-1), work.Off(givnum+st1-1).Matrix(n, opts), work.Off(c+st1-1), work.Off(s+st1-1), work.Off(nwork-1), toSlice(iwork, iwk-1)); err != nil {
					return
				}
			}
			st = i + 1
		}
	}

	//     Apply the singular values and treat the tiny ones as zero.
	tol = rcnd * math.Abs(d.Get(d.Iamax(n, 1)-1))

	for i = 1; i <= n; i++ {
		//        Some of the elements in D can be negative because 1-by-1
		//        subproblems were not solved explicitly.
		if math.Abs(d.Get(i-1)) <= tol {
			Dlaset(Full, 1, nrhs, zero, zero, work.Off(bx+i-1-1).Matrix(n, opts))
		} else {
			rank = rank + 1
			if err = Dlascl('G', 0, 0, d.Get(i-1), one, 1, nrhs, work.Off(bx+i-1-1).Matrix(n, opts)); err != nil {
				panic(err)
			}
		}
		d.Set(i-1, math.Abs(d.Get(i-1)))
	}

	//     Now apply back the right singular vectors.
	icmpq2 = 1
	for i = 1; i <= nsub; i++ {
		st = (*iwork)[i-1]
		st1 = st - 1
		nsize = (*iwork)[sizei+i-1-1]
		bxst = bx + st1
		if nsize == 1 {
			b.Off(st-1, 0).Vector().Copy(nrhs, work.Off(bxst-1), n, b.Rows)
		} else if nsize <= smlsiz {
			if err = b.Off(st-1, 0).Gemm(Trans, NoTrans, nsize, nrhs, nsize, one, work.Off(vt+st1-1).Matrix(n, opts), work.Off(bxst-1).Matrix(n, opts), zero); err != nil {
				panic(err)
			}
		} else {
			if err = Dlalsa(icmpq2, smlsiz, nsize, nrhs, work.Off(bxst-1).Matrix(n, opts), b.Off(st-1, 0), work.Off(u+st1-1).Matrix(n, opts), work.Off(vt+st1-1).Matrix(n, opts), toSlice(iwork, k+st1-1), work.Off(difl+st1-1).Matrix(n, opts), work.Off(difr+st1-1).Matrix(n, opts), work.Off(z+st1-1).Matrix(n, opts), work.Off(poles+st1-1).Matrix(n, opts), toSlice(iwork, givptr+st1-1), toSlice(iwork, givcol+st1-1), n, toSlice(iwork, perm+st1-1), work.Off(givnum+st1-1).Matrix(n, opts), work.Off(c+st1-1), work.Off(s+st1-1), work.Off(nwork-1), toSlice(iwork, iwk-1)); err != nil {
				return
			}
		}
	}

	//     Unscale and sort the singular values.
	if err = Dlascl('G', 0, 0, one, orgnrm, n, 1, d.Matrix(n, opts)); err != nil {
		panic(err)
	}
	if err = Dlasrt('D', n, d); err != nil {
		panic(err)
	}
	if err = Dlascl('G', 0, 0, orgnrm, one, n, nrhs, b); err != nil {
		panic(err)
	}

	return
}
