/**
 @file   DefDataBase.cc
 @author Yibo Lin
 @date   Dec 2014
 @brief  Implementation of @ref DefParser::DefDataBase
 */

#include "DefDataBase.h"
#include <cstring>
#include <cstdlib>

namespace DefParser {

void DefDataBase::set_def_diearea(int, const int*, const int*)
{
	def_user_cbk_reminder(__func__);
}

void DefDataBase::resize_def_blockage(int) 
{
	def_user_cbk_reminder(__func__);
}
void DefDataBase::add_def_placement_blockage(std::vector<std::vector<int> >const&)
{
	def_user_cbk_reminder(__func__);
}
void DefDataBase::resize_def_region(int)
{
	def_user_cbk_reminder(__func__);
}
void DefDataBase::add_def_region(Region const&)
{
	def_user_cbk_reminder(__func__);
}
void DefDataBase::resize_def_group(int)
{
	def_user_cbk_reminder(__func__);
}
void DefDataBase::add_def_group(Group const&)
{
	def_user_cbk_reminder(__func__);
}
void DefDataBase::end_def_design() 
{
	def_user_cbk_reminder(__func__);
}
void DefDataBase::def_user_cbk_reminder(const char* str) const 
{
	cout << "A corresponding user-defined callback is necessary: " << str << endl;
	exit(0);
}

void DefDataBase::add_def_track(defiTrack const&) {
	def_user_cbk_reminder(__func__);
}
void DefDataBase::add_def_gcellgrid(GCellGrid const&) {
	def_user_cbk_reminder(__func__);
}
void DefDataBase::add_def_snet(defiNet const&) {
	def_user_cbk_reminder(__func__);
}
void DefDataBase::add_def_via(defiVia const&) {
	def_user_cbk_reminder(__func__);
}
void DefDataBase::add_def_route_blockage(std::vector<std::vector<int> > const&, std::string const&) {
	def_user_cbk_reminder(__func__);
}
        

}
