package golapack

import (
	"math"

	"github.com/whipstein/golinalg/mat"
)

// Dlaein uses inverse iteration to find a right or left eigenvector
// corresponding to the eigenvalue (WR,WI) of a real upper Hessenberg
// matrix H.
func Dlaein(rightv, noinit bool, n int, h *mat.Matrix, wr, wi float64, vr, vi *mat.Vector, b *mat.Matrix, work *mat.Vector, eps3, smlnum, bignum float64) (info int) {
	var normin byte
	var trans mat.MatTrans
	var absbii, absbjj, ei, ej, growto, norm, nrmsml, one, rec, rootn, scale, temp, tenth, vcrit, vmax, vnorm, w, w1, x, xi, xr, y, zero float64
	var i, i1, i2, i3, its, j int
	var err error

	zero = 0.0
	one = 1.0
	tenth = 1.0e-1

	//     GROWTO is the threshold used in the acceptance test for an
	//     eigenvector.
	rootn = math.Sqrt(float64(n))
	growto = tenth / rootn
	nrmsml = math.Max(one, eps3*rootn) * smlnum

	//     Form B = H - (WR,WI)*I (except that the subdiagonal elements and
	//     the imaginary parts of the diagonal elements are not stored).
	for j = 1; j <= n; j++ {
		for i = 1; i <= j-1; i++ {
			b.Set(i-1, j-1, h.Get(i-1, j-1))
		}
		b.Set(j-1, j-1, h.Get(j-1, j-1)-wr)
	}

	if wi == zero {
		//        Real eigenvalue.
		if noinit {
			//           Set initial vector.
			for i = 1; i <= n; i++ {
				vr.Set(i-1, eps3)
			}
		} else {
			//           Scale supplied initial vector.
			vnorm = vr.Nrm2(n, 1)
			vr.Scal(n, (eps3*rootn)/math.Max(vnorm, nrmsml), 1)
		}

		if rightv {
			//           LU decomposition with partial pivoting of B, replacing zero
			//           pivots by EPS3.
			for i = 1; i <= n-1; i++ {
				ei = h.Get(i, i-1)
				if math.Abs(b.Get(i-1, i-1)) < math.Abs(ei) {
					//                 Interchange rows and eliminate.
					x = b.Get(i-1, i-1) / ei
					b.Set(i-1, i-1, ei)
					for j = i + 1; j <= n; j++ {
						temp = b.Get(i, j-1)
						b.Set(i, j-1, b.Get(i-1, j-1)-x*temp)
						b.Set(i-1, j-1, temp)
					}
				} else {
					//                 Eliminate without interchange.
					if b.Get(i-1, i-1) == zero {
						b.Set(i-1, i-1, eps3)
					}
					x = ei / b.Get(i-1, i-1)
					if x != zero {
						for j = i + 1; j <= n; j++ {
							b.Set(i, j-1, b.Get(i, j-1)-x*b.Get(i-1, j-1))
						}
					}
				}
			}
			if b.Get(n-1, n-1) == zero {
				b.Set(n-1, n-1, eps3)
			}

			trans = NoTrans

		} else {
			//           UL decomposition with partial pivoting of B, replacing zero
			//           pivots by EPS3.
			for j = n; j >= 2; j-- {
				ej = h.Get(j-1, j-1-1)
				if math.Abs(b.Get(j-1, j-1)) < math.Abs(ej) {
					//                 Interchange columns and eliminate.
					x = b.Get(j-1, j-1) / ej
					b.Set(j-1, j-1, ej)
					for i = 1; i <= j-1; i++ {
						temp = b.Get(i-1, j-1-1)
						b.Set(i-1, j-1-1, b.Get(i-1, j-1)-x*temp)
						b.Set(i-1, j-1, temp)
					}
				} else {
					//                 Eliminate without interchange.
					if b.Get(j-1, j-1) == zero {
						b.Set(j-1, j-1, eps3)
					}
					x = ej / b.Get(j-1, j-1)
					if x != zero {
						for i = 1; i <= j-1; i++ {
							b.Set(i-1, j-1-1, b.Get(i-1, j-1-1)-x*b.Get(i-1, j-1))
						}
					}
				}
			}
			if b.Get(0, 0) == zero {
				b.Set(0, 0, eps3)
			}

			trans = Trans

		}

		normin = 'N'
		for its = 1; its <= n; its++ {
			//           Solve U*x = scale*v for a right eigenvector
			//             or U**T*x = scale*v for a left eigenvector,
			//           overwriting x on v.
			if scale, err = Dlatrs(Upper, trans, NonUnit, normin, n, b, vr, scale, work); err != nil {
				panic(err)
			}
			normin = 'Y'

			//           Test for sufficient growth in the norm of v.
			vnorm = vr.Asum(n, 1)
			if vnorm >= growto*scale {
				goto label120
			}

			//           Choose new orthogonal starting vector and try again.
			temp = eps3 / (rootn + one)
			vr.Set(0, eps3)
			for i = 2; i <= n; i++ {
				vr.Set(i-1, temp)
			}
			vr.Set(n-its, vr.Get(n-its)-eps3*rootn)
		}

		//        Failure to find eigenvector in N iterations.
		info = 1

	label120:
		;

		//        Normalize eigenvector.
		i = vr.Iamax(n, 1)
		vr.Scal(n, one/math.Abs(vr.Get(i-1)), 1)
	} else {
		//        Complex eigenvalue.
		if noinit {
			//           Set initial vector.
			for i = 1; i <= n; i++ {
				vr.Set(i-1, eps3)
				vi.Set(i-1, zero)
			}
		} else {
			//           Scale supplied initial vector.
			norm = Dlapy2(vr.Nrm2(n, 1), vi.Nrm2(n, 1))
			rec = (eps3 * rootn) / math.Max(norm, nrmsml)
			vr.Scal(n, rec, 1)
			vi.Scal(n, rec, 1)
		}

		if rightv {
			//           LU decomposition with partial pivoting of B, replacing zero
			//           pivots by EPS3.
			//
			//           The imaginary part of the (i,j)-th element of U is stored in
			//           B(j+1,i).
			b.Set(1, 0, -wi)
			for i = 2; i <= n; i++ {
				b.Set(i, 0, zero)
			}

			for i = 1; i <= n-1; i++ {
				absbii = Dlapy2(b.Get(i-1, i-1), b.Get(i, i-1))
				ei = h.Get(i, i-1)
				if absbii < math.Abs(ei) {
					//                 Interchange rows and eliminate.
					xr = b.Get(i-1, i-1) / ei
					xi = b.Get(i, i-1) / ei
					b.Set(i-1, i-1, ei)
					b.Set(i, i-1, zero)
					for j = i + 1; j <= n; j++ {
						temp = b.Get(i, j-1)
						b.Set(i, j-1, b.Get(i-1, j-1)-xr*temp)
						b.Set(j, i, b.Get(j, i-1)-xi*temp)
						b.Set(i-1, j-1, temp)
						b.Set(j, i-1, zero)
					}
					b.Set(i+2-1, i-1, -wi)
					b.Set(i, i, b.Get(i, i)-xi*wi)
					b.Set(i+2-1, i, b.Get(i+2-1, i)+xr*wi)
				} else {
					//                 Eliminate without interchanging rows.
					if absbii == zero {
						b.Set(i-1, i-1, eps3)
						b.Set(i, i-1, zero)
						absbii = eps3
					}
					ei = (ei / absbii) / absbii
					xr = b.Get(i-1, i-1) * ei
					xi = -b.Get(i, i-1) * ei
					for j = i + 1; j <= n; j++ {
						b.Set(i, j-1, b.Get(i, j-1)-xr*b.Get(i-1, j-1)+xi*b.Get(j, i-1))
						b.Set(j, i, -xr*b.Get(j, i-1)-xi*b.Get(i-1, j-1))
					}
					b.Set(i+2-1, i, b.Get(i+2-1, i)-wi)
				}

				//              Compute 1-norm of offdiagonal elements of i-th row.
				work.Set(i-1, b.Off(i-1, i).Vector().Asum(n-i, b.Rows)+b.Off(i+2-1, i-1).Vector().Asum(n-i, 1))
			}
			if b.Get(n-1, n-1) == zero && b.Get(n, n-1) == zero {
				b.Set(n-1, n-1, eps3)
			}
			work.Set(n-1, zero)

			i1 = n
			i2 = 1
			i3 = -1
		} else {
			//           UL decomposition with partial pivoting of conjg(B),
			//           replacing zero pivots by EPS3.
			//
			//           The imaginary part of the (i,j)-th element of U is stored in
			//           B(j+1,i).
			b.Set(n, n-1, wi)
			for j = 1; j <= n-1; j++ {
				b.Set(n, j-1, zero)
			}

			for j = n; j >= 2; j-- {
				ej = h.Get(j-1, j-1-1)
				absbjj = Dlapy2(b.Get(j-1, j-1), b.Get(j, j-1))
				if absbjj < math.Abs(ej) {
					//                 Interchange columns and eliminate
					xr = b.Get(j-1, j-1) / ej
					xi = b.Get(j, j-1) / ej
					b.Set(j-1, j-1, ej)
					b.Set(j, j-1, zero)
					for i = 1; i <= j-1; i++ {
						temp = b.Get(i-1, j-1-1)
						b.Set(i-1, j-1-1, b.Get(i-1, j-1)-xr*temp)
						b.Set(j-1, i-1, b.Get(j, i-1)-xi*temp)
						b.Set(i-1, j-1, temp)
						b.Set(j, i-1, zero)
					}
					b.Set(j, j-1-1, wi)
					b.Set(j-1-1, j-1-1, b.Get(j-1-1, j-1-1)+xi*wi)
					b.Set(j-1, j-1-1, b.Get(j-1, j-1-1)-xr*wi)
				} else {
					//                 Eliminate without interchange.
					if absbjj == zero {
						b.Set(j-1, j-1, eps3)
						b.Set(j, j-1, zero)
						absbjj = eps3
					}
					ej = (ej / absbjj) / absbjj
					xr = b.Get(j-1, j-1) * ej
					xi = -b.Get(j, j-1) * ej
					for i = 1; i <= j-1; i++ {
						b.Set(i-1, j-1-1, b.Get(i-1, j-1-1)-xr*b.Get(i-1, j-1)+xi*b.Get(j, i-1))
						b.Set(j-1, i-1, -xr*b.Get(j, i-1)-xi*b.Get(i-1, j-1))
					}
					b.Set(j-1, j-1-1, b.Get(j-1, j-1-1)+wi)
				}

				//              Compute 1-norm of offdiagonal elements of j-th column.
				work.Set(j-1, b.Off(0, j-1).Vector().Asum(j-1, 1)+b.Off(j, 0).Vector().Asum(j-1, b.Rows))
			}
			if b.Get(0, 0) == zero && b.Get(1, 0) == zero {
				b.Set(0, 0, eps3)
			}
			work.Set(0, zero)

			i1 = 1
			i2 = n
			i3 = 1
		}

		for its = 1; its <= n; its++ {
			scale = one
			vmax = one
			vcrit = bignum

			//           Solve U*(xr,xi) = scale*(vr,vi) for a right eigenvector,
			//             or U**T*(xr,xi) = scale*(vr,vi) for a left eigenvector,
			//           overwriting (xr,xi) on (vr,vi).
			for _, i = range genIter(i1, i2, i3) {

				if work.Get(i-1) > vcrit {
					rec = one / vmax
					vr.Scal(n, rec, 1)
					vi.Scal(n, rec, 1)
					scale = scale * rec
					vmax = one
					vcrit = bignum
				}

				xr = vr.Get(i - 1)
				xi = vi.Get(i - 1)
				if rightv {
					for j = i + 1; j <= n; j++ {
						xr = xr - b.Get(i-1, j-1)*vr.Get(j-1) + b.Get(j, i-1)*vi.Get(j-1)
						xi = xi - b.Get(i-1, j-1)*vi.Get(j-1) - b.Get(j, i-1)*vr.Get(j-1)
					}
				} else {
					for j = 1; j <= i-1; j++ {
						xr = xr - b.Get(j-1, i-1)*vr.Get(j-1) + b.Get(i, j-1)*vi.Get(j-1)
						xi = xi - b.Get(j-1, i-1)*vi.Get(j-1) - b.Get(i, j-1)*vr.Get(j-1)
					}
				}

				w = math.Abs(b.Get(i-1, i-1)) + math.Abs(b.Get(i, i-1))
				if w > smlnum {
					if w < one {
						w1 = math.Abs(xr) + math.Abs(xi)
						if w1 > w*bignum {
							rec = one / w1
							vr.Scal(n, rec, 1)
							vi.Scal(n, rec, 1)
							xr = vr.Get(i - 1)
							xi = vi.Get(i - 1)
							scale = scale * rec
							vmax = vmax * rec
						}
					}

					//                 Divide by diagonal element of B.
					_vr := vr.GetPtr(i - 1)
					_vi := vi.GetPtr(i - 1)
					*_vr, *_vi = Dladiv(xr, xi, b.Get(i-1, i-1), b.Get(i, i-1))
					vmax = math.Max(math.Abs(vr.Get(i-1))+math.Abs(vi.Get(i-1)), vmax)
					vcrit = bignum / vmax
				} else {
					for j = 1; j <= n; j++ {
						vr.Set(j-1, zero)
						vi.Set(j-1, zero)
					}
					vr.Set(i-1, one)
					vi.Set(i-1, one)
					scale = zero
					vmax = one
					vcrit = bignum
				}
			}

			//           Test for sufficient growth in the norm of (VR,VI).
			vnorm = vr.Asum(n, 1) + vi.Asum(n, 1)
			if vnorm >= growto*scale {
				goto label280
			}

			//           Choose a new orthogonal starting vector and try again.
			y = eps3 / (rootn + one)
			vr.Set(0, eps3)
			vi.Set(0, zero)

			for i = 2; i <= n; i++ {
				vr.Set(i-1, y)
				vi.Set(i-1, zero)
			}
			vr.Set(n-its, vr.Get(n-its)-eps3*rootn)
		}

		//        Failure to find eigenvector in N iterations
		info = 1

	label280:
		;

		//        Normalize eigenvector.
		vnorm = zero
		for i = 1; i <= n; i++ {
			vnorm = math.Max(vnorm, math.Abs(vr.Get(i-1))+math.Abs(vi.Get(i-1)))
		}
		vr.Scal(n, one/vnorm, 1)
		vi.Scal(n, one/vnorm, 1)

	}

	return
}
